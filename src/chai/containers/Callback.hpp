#ifndef CHAI_CALLBACK_HPP
#define CHAI_CALLBACK_HPP

namespace chai {
   class MemoryManagementCallback {
      template <class ActualCallback>
      Callback(

      template <class ExecutionManager>
      void operator()(const ExecutionManager& execution_manager);

      template <class ExecutionEvent>
      void operator()(const ExecutionEvent& execution_event);
   };
}  // namespace chai

#endif  // CHAI_CALLBACK_HPP
