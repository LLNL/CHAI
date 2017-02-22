namespace sump {

class ResourceManager {
  ResourceManager* getResourceManager();
 

  protected:
    ResourceManager();

  private: 
    ResourceManager* s_resource_manager_instance;
}

}
