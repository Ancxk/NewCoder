package 设计模式.strary.strategyImpl;

import 设计模式.strary.CreateVideoStrategy;

/**
 * @author xwp
 * @date 2024/5/24
 * @Description
 */
public class TikTokImpl implements CreateVideoStrategy {
    @Override
    public void uploadVideo(String url) {
        try {
            System.out.println("tk");
        }catch (Exception ignored){


        }

    }

    @Override
    public void createVideo(String type) {
        System.out.println("tk");
    }

    @Override
    public void pushVideo(String meta) {
        System.out.println("tk");
    }
}
