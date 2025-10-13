//===- equiv_miter.cpp - The equiv_miter driver ---------------------*- C++ -*-===//

#include "circt/Conversion/CombToSMT.h"
#include "circt/Conversion/HWToSMT.h"
#include "circt/Conversion/SMTToZ3LLVM.h"
#include "circt/Conversion/VerifToSMT.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Emit/EmitDialect.h"
#include "circt/Dialect/Emit/EmitPasses.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/OM/OMPasses.h"
#include "circt/Dialect/Verif/VerifDialect.h"
#include "circt/Dialect/AIG/AIGPasses.h"
#include "circt/Support/Passes.h"
#include "circt/Support/Version.h"
#include "mlir/Dialect/SMT/IR/SMTDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/SMTLIB/ExportSMTLIB.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"


#include "circt-passes/EquivMiter/Passes.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Conversion/CombToAIG.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Conversion/ExportAIGER.h"
#include "circt/Conversion/HWToBTOR2.h"

namespace cl = llvm::cl;

using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// Command-line options declaration
//===----------------------------------------------------------------------===//

static cl::OptionCategory mainCategory("equiv_miter Options");

static cl::opt<std::string> firstModuleName(
    "c1", cl::Required,
    cl::desc("Specify a named module for the first circuit of the comparison"),
    cl::value_desc("module name"), cl::cat(mainCategory));

static cl::opt<std::string> secondModuleName(
    "c2", cl::Required,
    cl::desc("Specify a named module for the second circuit of the comparison"),
    cl::value_desc("module name"), cl::cat(mainCategory));

static cl::list<std::string> inputFilenames(cl::Positional, cl::OneOrMore,
                                            cl::desc("<input files>"),
                                            cl::cat(mainCategory));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"),
                                           cl::cat(mainCategory));

static cl::opt<bool>
    verifyPasses("verify-each",
                 cl::desc("Run the verifier after each transformation pass"),
                 cl::init(true), cl::cat(mainCategory));

static cl::opt<bool>
    verbosePassExecutions("verbose-pass-executions",
                          cl::desc("Log executions of toplevel module passes"),
                          cl::init(false), cl::cat(mainCategory));

enum OutputFormat { OutputSMTLIB, OutputAIGER, OutputBTOR2 };
static cl::opt<OutputFormat> outputFormat(
    cl::desc("Specify output format"),
    cl::values(clEnumValN(OutputSMTLIB, "smtlib", "smt object file"),
               clEnumValN(OutputAIGER,  "aiger",  "aiger object file"),
               clEnumValN(OutputBTOR2,  "btor2",  "btor2 object file")),
    cl::init(OutputSMTLIB), cl::cat(mainCategory));


//===----------------------------------------------------------------------===//
// Tool implementation
//===----------------------------------------------------------------------===//

// Move all operations in `src` to `dest`. Rename all symbols in `src` to avoid
// conflict.
static FailureOr<StringAttr> mergeModules(ModuleOp dest, ModuleOp src,
                                          StringAttr name) {

  SymbolTable destTable(dest), srcTable(src);
  StringAttr newName = {};
  for (auto &op : src.getOps()) {
    if (SymbolOpInterface symbol = dyn_cast<SymbolOpInterface>(op)) {
      auto oldSymbol = symbol.getNameAttr();
      auto result = srcTable.renameToUnique(&op, {&destTable});
      if (failed(result))
        return src->emitError() << "failed to rename symbol " << oldSymbol;

      if (oldSymbol == name) {
        assert(!newName && "symbol must be unique");
        newName = *result;
      }
    }
  }

  if (!newName)
    return src->emitError()
           << "module " << name << " was not found in the second module";

  dest.getBody()->getOperations().splice(dest.getBody()->begin(),
                                         src.getBody()->getOperations());
  return newName;
}

// Parse one or two MLIR modules and merge it into a single module.
static FailureOr<OwningOpRef<ModuleOp>>
parseAndMergeModules(MLIRContext &context, TimingScope &ts) {
  auto parserTimer = ts.nest("Parse and merge MLIR input(s)");

  if (inputFilenames.size() > 2) {
    llvm::errs() << "more than 2 files are provided!\n";
    return failure();
  }

  auto module = parseSourceFile<ModuleOp>(inputFilenames[0], &context);
  if (!module)
    return failure();

  if (inputFilenames.size() == 2) {
    auto moduleOpt = parseSourceFile<ModuleOp>(inputFilenames[1], &context);
    if (!moduleOpt)
      return failure();
    auto result = mergeModules(module.get(), moduleOpt.get(),
                               StringAttr::get(&context, secondModuleName));
    if (failed(result))
      return failure();

    secondModuleName.setValue(result->getValue().str());
  }

  return module;
}

static LogicalResult executeMiterToSMTLIB(mlir::PassManager &pm, ModuleOp module,
                                          llvm::raw_ostream &os) {
    pm.addPass(om::createStripOMPass());
    pm.addPass(emit::createStripEmitPass());

    pm.addPass(hw::createFlattenModules());
    EquivfusionMiterOptions opts = {firstModuleName, secondModuleName, EquivMiter::MiterModeEnum::SMTLIB};
    pm.addPass(createEquivfusionMiter(opts));

    pm.addPass(createConvertHWToSMT());
    pm.addPass(createConvertCombToSMT());
    pm.addPass(createConvertVerifToSMT());

    pm.addPass(createSimpleCanonicalizerPass());

    if (failed(pm.run(module)))
        return failure();
    
    return smt::exportSMTLIB(module, os);
}

static LogicalResult executeMiterToAIGER(mlir::PassManager &pm, ModuleOp module,
                                         llvm::raw_ostream &os) {
    EquivfusionMiterOptions opts = {firstModuleName, secondModuleName, EquivMiter::MiterModeEnum::AIGER};
    pm.addPass(createEquivfusionMiter(opts));

    pm.addPass(hw::createFlattenModules());
    pm.addPass(createSimpleCanonicalizerPass());

    pm.nest<hw::HWModuleOp>().addPass(createConvertCombToAIG());
    pm.nest<hw::HWModuleOp>().addPass(aig::createLowerVariadic());

    if (failed(pm.run(module)))
        return failure();
    auto ops = module.getOps<hw::HWModuleOp>();
    if (ops.empty() || std::next(ops.begin()) != ops.end())
        return failure();

    return aiger::exportAIGER(*ops.begin(), os);
}


static LogicalResult executeMiterToBTOR2(mlir::PassManager &pm, ModuleOp module,
                                         llvm::raw_ostream &os) {
    EquivfusionMiterOptions opts = {firstModuleName, secondModuleName, EquivMiter::MiterModeEnum::BTOR2};
    pm.addPass(createEquivfusionMiter(opts));

    pm.addPass(hw::createFlattenModules());
    pm.addPass(createSimpleCanonicalizerPass());

    pm.addPass(createConvertHWToBTOR2Pass(os));

    return pm.run(module);
}


static LogicalResult executeMiter(MLIRContext &context) {
    // Create the timing manager we use to sample execution times.
    DefaultTimingManager tm;
    applyDefaultTimingManagerCLOptions(tm);
    auto ts = tm.getRootScope();

    auto parsedModule = parseAndMergeModules(context, ts);
    if (failed(parsedModule))
        return failure();

    OwningOpRef<ModuleOp> module = std::move(parsedModule.value());

    // Create the output directory or output file depending on our mode.
    std::optional<std::unique_ptr<llvm::ToolOutputFile>> outputFile;
    std::string errorMessage;
    // Create an output file.
    outputFile.emplace(openOutputFile(outputFilename, &errorMessage));
    if (!outputFile.value()) {
        llvm::errs() << errorMessage << "\n";
        return failure();
    }

    PassManager pm(&context);
    pm.enableVerifier(verifyPasses);
    pm.enableTiming(ts);
    if (failed(applyPassManagerCLOptions(pm)))
        return failure();

    if (verbosePassExecutions)
        pm.addInstrumentation(
            std::make_unique<VerbosePassInstrumentation<mlir::ModuleOp>>(
                "equiv_miter"));

    LogicalResult result = failure();
    switch (outputFormat) {
    case OutputSMTLIB:
        result = executeMiterToSMTLIB(pm, module.get(), outputFile.value()->os());
        break;
    case OutputAIGER:
        result = executeMiterToAIGER(pm, module.get(), outputFile.value()->os());
        break;
    case OutputBTOR2:
        result = executeMiterToBTOR2(pm, module.get(), outputFile.value()->os());
        break;
    }

    if (failed(result))
        return failure();

    outputFile.value()->keep();
    return success();
}

/// The entry point for the `equiv_miter` tool:
/// configures and parses the command-line options,
/// registers all dialects within a MLIR context,
/// and calls the `executeLEC` function to do the actual work.
int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Hide default LLVM options, other than for this tool.
  // MLIR options are added below.
  cl::HideUnrelatedOptions(mainCategory);

  // Register any pass manager command line options.
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerAsmPrinterCLOptions();
  cl::AddExtraVersionPrinter(
      [](llvm::raw_ostream &os) { os << circt::getCirctVersion() << '\n'; });

  // Parse the command-line options provided by the user.
  cl::ParseCommandLineOptions(
      argc, argv,
      "equiv_miter - construct Miter\n\n"
      "\tThis tool construct miter for two input circuit descriptions.\n");

  // Set the bug report message to indicate users should file issues on
  // llvm/circt and not llvm/llvm-project.
  llvm::setBugReportMsg(circt::circtBugReportMsg);

  // Register the supported CIRCT dialects and create a context to work with.
  DialectRegistry registry;
  // clang-format off
  registry.insert<
    circt::comb::CombDialect,
    circt::emit::EmitDialect,
    circt::hw::HWDialect,
    circt::om::OMDialect,
    mlir::smt::SMTDialect,
    circt::verif::VerifDialect,
    mlir::func::FuncDialect
  >();  
  // clang-format on
  mlir::func::registerInlinerExtension(registry);
  MLIRContext context(registry);

  // Setup of diagnostic handling.
  llvm::SourceMgr sourceMgr;
  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
  // Avoid printing a superfluous note on diagnostic emission.
  context.printOpOnDiagnostic(false);

  // Perform the logical equivalence checking; using `exit` to avoid the slow
  // teardown of the MLIR context.
  exit(failed(executeMiter(context)));
}
