#ifndef CHAI_context_manager_HPP
#define CHAI_context_manager_HPP

#include "chai/ExecutionSpaces.hpp"

class context_manager
{
   bool hasContext() const;
   ExecutionSpace getContext;
   void pushContext(ExecutionSpace);
   void popContext();
};

extern context_manager* g_default_context;

#endif // CHAI_context_manager_HPP