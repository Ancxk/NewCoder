package 设计模式.strary;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

/**
 * @author xwp
 * @date 2024/5/24
 * @Description
 */
public abstract class AbstractCreateVideo {

    //依赖注入
    private final Map<String,CreateVideoStrategy> videoStrategyMap = new HashMap<>();

    public void MustPostVideo(String url,String type){
        CreateVideoStrategy createVideoStrategy = videoStrategyMap.get(type);
        createVideoStrategy.createVideo(url);
        createVideoStrategy.uploadVideo(url);
        createVideoStrategy.pushVideo(url);
    }
}
