package 设计模式.strary;

/**
 * @author xwp
 * @date 2024/5/24
 * @Description
 */
public interface CreateVideoStrategy {
    void uploadVideo(String url);
    void createVideo(String type);
    void pushVideo(String meta);
}
