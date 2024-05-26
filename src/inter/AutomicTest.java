package inter;

import java.util.concurrent.*;

/**
 * @author xwp
 * @date 2024/5/20
 * @Description
 */
public class AutomicTest {
    public static void main(String[] args) {
        Thread hello = new Thread(() -> System.out.println("hello"));
        hello.start();
//        ThreadPoolExecutor threadPoolExecutor = new ThreadPoolExecutor(2,10,10L, TimeUnit.SECONDS);


        ExecutorService threadPool = Executors.newFixedThreadPool(2);
        FutureTask<Integer> task = new FutureTask<>(() -> {
            System.out.println("hello");
            return 1;
        });
        threadPool.submit(task);
    }

}
