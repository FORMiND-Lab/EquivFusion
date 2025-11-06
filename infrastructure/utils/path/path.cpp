#include "infrastructure/utils/path/path.h"

namespace XuanSong {
namespace Utils {
namespace PathUtil {

void expandTilde(std::string& path) {
    if (path.compare(0, 2, "~/") == 0)
        path = path.replace(0, 1, getenv("HOME"));
}

} // namespace PathUtil
} // namespace Utils
} // namespace XuanSong
