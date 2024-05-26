package 设计模式.strary.strategyImpl;

import 设计模式.strary.CreateVideoStrategy;

/**
 * @author xwp
 * @date 2024/5/24
 * @Description
 */
public class YoutubeImpl implements CreateVideoStrategy {
    @Override
    public void uploadVideo(String url) {
        System.out.println("youtube");
    }

    @Override
    public void createVideo(String type) {
        System.out.println("youtube");
    }

    @Override
    public void pushVideo(String meta) {
        System.out.println("youtube");
    }
}
