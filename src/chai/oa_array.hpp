#ifndef CHAI_oa_array_HPP
#define CHAI_oa_array_HPP

namespace chai {

template<typename TTT>
class oa_array {
 public:
   //default constructor
   oa_array();

   //manual destructor
   ~oa_array();

   //copy constructors with copy and swap idiom
   oa_array(const oa_array<TTT>& other);
   oa_array(oa_array<TTT>&& other);
   oa_array<TTT>& operator=(oa_array<TTT> other);
   friend inline void swap(oa_array<TTT>&, oa_array<TTT>&);

   TTT* use_on(const ExecutionSpace);
   oa_array<TTT> slice(const std::size_t begin, const std::size_t end);

   template <typename Index>
   inline OnlyAssignable<TTT&> operator[](const IndexWrapper<Index>);
   
   std::size_t size() const;
};

} // end of namespace chai

#endif // CHAI_oa_array_HPP
