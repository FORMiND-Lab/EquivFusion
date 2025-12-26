#ifndef EQUIVFUSION_READ_FIRRTL_H
#define EQUIVFUSION_READ_FIRRTL_H

#include <vector>
#include <string>
#include "infrastructure/utils/namespace_macro.h"

XUANSONG_NAMESPACE_HEADER_START
namespace ReadFIRRTLTool {

void help(const std::string &name, const std::string &description);

bool execute(const std::vector<std::string> &args);

} // namespace ReadFIRRTLTool
XUANSONG_NAMESPACE_HEADER_END // namespace XuanSong

#endif // EQUIVFUSION_READ_FIRRTL_H
