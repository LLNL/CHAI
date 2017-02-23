namespace umpire {

class ResourceManager {
  public:
    ResourceManager* getResourceManager();

    size_t getAvailableMemory();


  protected:
    ResourceManager();

  private: 
    ResourceManager* s_resource_manager_instance;
};

}
