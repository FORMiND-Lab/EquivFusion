#ifndef EQUIVFUSION_EQUIV_MITER_TOOL_H
#define EQUIVFUSION_EQUIV_MITER_TOOL_H

#include "infrastructure/utils/namespace_macro.h"

#include "llvm/Support/CommandLine.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/CommandLine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/Timing.h"

namespace cl = llvm::cl;
using namespace mlir;

XUANSONG_NAMESPACE_HEADER_START

class EquivMiterTool {
public:
    EquivMiterTool() = default;

public:
    int run(int argc, char **argv);
    int run(const std::vector<std::string> &args)

private:
    FailureOr<OwningOpRef<ModuleOp>> parseAndMergeModules(MLIRContext &context, TimingScope &ts);
    FailureOr<StringAttr> mergeModules(ModuleOp dest, ModuleOp src, StringAttr name);

private:
    LogicalResult executeMiter(MLIRContext &context, OwningOpRef<ModuleOp>& module, mlir::TimingScope &ts);
    LogicalResult executeMiterToSMTLIB(mlir::PassManager &pm, ModuleOp module, llvm::raw_ostream &os);
    LogicalResult executeMiterToAIGER(mlir::PassManager &pm, ModuleOp module, llvm::raw_ostream &os);
    LogicalResult executeMiterToBTOR2(mlir::PassManager &pm, ModuleOp module, llvm::raw_ostream &os);


private:
    // options
    cl::OptionCategory mainCategory{"equiv_miter Options"};

    cl::opt<std::string> firstModuleName{
        "c1", cl::Required,
        cl::desc("Specify a named module for the first circuit of the comparison"),
        cl::value_desc("module name"), cl::cat(mainCategory)};

    cl::opt<std::string> secondModuleName{
        "c2", cl::Required,
        cl::desc("Specify a named module for the second circuit of the comparison"),
        cl::value_desc("module name"), cl::cat(mainCategory)};

    cl::list<std::string> inputFilenames{cl::Positional, cl::OneOrMore,
                                         cl::desc("<input files>"),
                                         cl::cat(mainCategory)};

    cl::opt<std::string> outputFilename{"o", cl::desc("Output filename"),
                                        cl::value_desc("filename"),
                                        cl::init("-"),
                                        cl::cat(mainCategory)};

    enum OutputFormat { OutputSMTLIB, OutputAIGER, OutputBTOR2 };
    cl::opt<OutputFormat> outputFormat{
        cl::desc("Specify output format"),
        cl::values(clEnumValN(OutputSMTLIB, "smtlib", "smt object file"),
                   clEnumValN(OutputAIGER,  "aiger",  "aiger object file"),
                   clEnumValN(OutputBTOR2,  "btor2",  "btor2 object file")),
        cl::init(OutputSMTLIB), cl::cat(mainCategory)};
};

XUANSONG_NAMESPACE_HEADER_END // namespace XuanSong

#endif //EQUIVFUSION_EQUIV_MITER_TOOL_H
