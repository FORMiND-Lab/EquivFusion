//===- equiv_miter.cpp - The equiv_miter driver ---------------------*- C++ -*-===//

#include "libs/Tools/EquivMiterTool/equiv_miter_tool.h"

int main(int argc, char **argv) {
    XuanSong::EquivMiterTool equivMiterTool;
    exit(equivMiterTool.run(argc, argv));
}
