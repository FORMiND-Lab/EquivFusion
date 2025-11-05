#ifndef EQUIVFUSION_CONVERT_CASE_TO_LOGICAL_COMPARSION
#define EQUIVFUSION_CONVERT_CASE_TO_LOGICAL_COMPARSION

include "mlir/Pass/PassBase.td"

def EquivFusionConvertCaseToLogicalComparsionPass : Pass<"equivfusion-convert-case-to-logical-comparsion", "mlir::ModuleOp"> { 
    let summary = "Convert the predicate of 'comb.icmp' from 'ceq/cne' to 'eq/ne'.";

    let description = [{
        This pass converts the predicate of 'comb.icmp' from 'ceq/cne' to 'eq/ne'.
    }];

    let dependentDialects = ["circt::comb::CombDialect"];
}



#endif // EQUIVFUSION_CONVERT_CASE_TO_LOGICAL_COMPARSION

