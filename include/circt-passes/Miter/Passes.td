#ifndef EQUIVFUSION_MITER_PASSES_TD
#define EQUIVFUSION_MITER_PASSES_TD

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def EquivFusionMiter : Pass<"EquivFusion-miter", "::mlir::ModuleOp"> {
    let summary = "Lower CIRCTs core dialects to a LEC problem statement";
    let description = [{
        Taks two `hw.module` operations and lowers them to `smt` or `aiger` or `btor2`.
    }];

    let options = [
        Option<"firstModule", "first-module", "std::string",
               /*default=*/"",
               "Name of the first of the two modules to compare.">,
        Option<"secondModule", "second-module", "std::string",
               /*default=*/"",
               "Name of the second of the two modules to compare.">,
        Option<"miterMode", "miter-mode", "MiterModeEnum",
              /*default=*/"MiterModeEnum::SMTLIB",
            "Select what additional code to add.",
            [{::llvm::cl::values(
             clEnumValN(MiterModeEnum::SMTLIB,
             "SMTLIB", "Miter for SMTLIB output."),
             clEnumValN(MiterModeEnum::AIGER,
             "AIGER", "Miter for AIGER output."),
             clEnumValN(MiterModeEnum::BTOR2,
             "BTOR2", "Miter for BTOR2 output.")
           )}]>,
    ];

    let dependentDialects = [
        "verif::VerifDialect",
    ];
}

#endif // EQUIVFUSION_MITER_PASSES_TD
