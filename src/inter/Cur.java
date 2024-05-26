package inter;

import java.util.Map;
import java.util.concurrent.ConcurrentSkipListMap;
import java.util.concurrent.ConcurrentSkipListSet;
import java.util.function.Consumer;

/**
 * @author xwp
 * @date 2024/5/4
 * @Description
 */
public class Cur {
    public static void main(String[] args) {
        ConcurrentSkipListSet<Integer> skipSet = new ConcurrentSkipListSet<>();
        skipSet.add(1);
        skipSet.add(2);
        skipSet.add(3);
        Map<Integer, Integer> skipMap = new ConcurrentSkipListMap<>();
        skipMap.put(1,2);
        skipMap.put(2,3);
        skipMap.put(3,4);
        Integer ceiling = skipSet.ceiling(1);
        System.out.println(ceiling);
    }
}





