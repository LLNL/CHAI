#ifndef CHAI_ro_array_HPP
#define CHAI_ro_array_HPP

namespace chai {

template<typename TTT>
class ro_array {
 public:
   //default constructor
   ro_array();

   //manual destructor
   ~ro_array();

   //copy constructors with copy and swap idiom
   ro_array(const ro_array<TTT>& other);
   ro_array(ro_array<TTT>&& other);
   ro_array<TTT>& operator=(ro_array<TTT> other);
   friend inline void swap(ro_array<TTT>&, ro_array<TTT>&);   
   
   const TTT* use_on(const ExecutionSpace) const;
   ro_array<TTT> slice(const std::size_t begin, const std::size_t end) const;

   template <typename Index>
   inline IndexWrapper<const TTT&> operator[](const IndexWrapper<Index>) const;
   
   std::size_t size() const;
};

} // end of namespace chai

#endif CHAI_ro_array_HPP
