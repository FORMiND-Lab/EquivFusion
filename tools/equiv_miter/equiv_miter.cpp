//===- equiv_miter.cpp - The equiv_miter driver ---------------------*- C++ -*-===//

#include "infrastructure/tools/EquivMiterTool/equiv_miter_tool.h"

int main(int argc, char **argv) {
    EquivMiterTool equivMiterTool;
    exit(equivMiterTool.run(argc, argv));
}
