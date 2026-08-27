# 上游与只读依赖锁定

- Sparse-VideoGen commit：`f89aedaf169ac2ae5b186bda674e53c3dc08c476`
- SVOO commit：`e4ae67b579766bcbe820bda7d34e104ff4c82d5f`
- 官方Wan T2V 720p SAP参考：Q=300、K=1000、Top-p=0.9、min-K=0.1、init=50、step=2；对应14B、1280×720，不等同于本项目1.3B、832×480。
- vendored `svoo/co_clustering.py` SHA-256：`8b2d1e52b7151b4d763f3ac93c7759df21e03b2f4785457a56fa2252ef6af2fb`
- vendored permutation SHA-256：`ad4d16c114fd68fab15b9f2206f47d21db12674a8ef6b15e764ba7cedacd6ce2`
- fixed64 interface SHA-256：`0d73856422a575280b17d59211a041aac15af7984cb9db4efb3831ea6c00eb06`
- fixed64 Triton kernel SHA-256：`3282cb9ac56051086b9631b230ce53a3b4b66904fc7978e00a1953aa476c37b3`
- Wan1.3B transformer config SHA-256：`0b093fa072e9ff28763febe9b964ee582f566733a6d6709deb9dfba1bde16b81`

0820项目、模型目录和`fp8-sparse-attn`均为只读依赖；新代码、缓存和结果只写本工作流目录。许可证副本保存在`adapters/vendor/licenses/`。
