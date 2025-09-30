#ifndef SOLVER_HPP
#define SOLVER_HPP

#include <string>
#include "solving/solver_definitions.h"

namespace XuanSong {

class Solver {
public:
    virtual ~Solver() = default;
    virtual int run(const std::string &command) = 0;
    virtual std::string getName() const = 0;

protected:
    int executeCommand(const std::string &command) {
        return system(command.c_str());
    }
};

// Declare Solver Class
#define SOLVER(ClassName, SolverName) \
class ClassName : public Solver { \
public: \
    std::string getName() const override { return SolverName; } \
    int run(const std::string &command) override; \
};

SOLVER_LIST
#undef SOLVER


} // namespace XuanSong

#endif //SOLVER_HPP


