#ifndef CHAI_wo_array_HPP
#define CHAI_wo_array_HPP

namespace chai {
template<typename TTT>
class wo_array {
 public:
   //default constructor
   wo_array();

   //manual destructor
   ~wo_array();
   
   //copy constructors with copy and swap idiom
   wo_array(const wo_array<TTT>& other);
   wo_array(wo_array<TTT>&& other);
   wo_array<TTT>& operator=(wo_array<TTT> other);
   friend inline void swap(wo_array<TTT>&, wo_array<TTT>&);

   TTT* use_on(const ExecutionSpace);
   wo_array<TTT> slice(const std::size_t begin, const std::size_t end);

   template <typename Index>
   inline OnlyAssignable<TTT&> operator[](const IndexWrapper<Index>);
   
   std::size_t size() const;

   oa_array<TTT> overwiteall();
   TTT* overwiteall(ExecutionSpace);
   inline operator oa_array<TTT>();
};

} // end of namespace chai

#endif // CHAI_wo_array_HPP
