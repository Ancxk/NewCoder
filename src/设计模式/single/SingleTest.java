package 设计模式.single;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * @author xwp
 * @date 2024/5/25
 * @Description
 */
public class SingleTest {
    public static void main(String[] args) {
        ThreadLocal<Integer> threadLocal = new ThreadLocal<>();
        ThreadLocal<String> threadLocal2 = new ThreadLocal<>();
        threadLocal.set(4);
        threadLocal.set(5);
        Integer integer = threadLocal.get();
        System.out.println(integer);
        threadLocal2.set("adw");


//        int cores = Runtime.getRuntime().availableProcessors();
//
//        // 打印核心数
//        System.out.println("Available CPU cores: " + cores);
//        ExecutorService threadPool = Executors.newFixedThreadPool(10000);
//        int i1 = Single.getSingle().hashCode();
//
//        for (int i = 0; i < 10000; i++) {
//            threadPool.submit(()->{
//                Single single = Single.getSingle();
//                System.out.println(single.hashCode());
//                try {
//                    Thread.sleep(500L);
//                } catch (InterruptedException e) {
//                    e.printStackTrace();
//                }
//                if(i1 != single.hashCode()){
//                    throw new RuntimeException("replicate");
//                }
//            });
//        }
//        threadPool.shutdown();
//    }
    }
}


class Single{
    private volatile static  Single single = null;
    public static Single getSingle(){
        if (single == null){
            synchronized (Single.class){
                if(single == null){
                    single = new Single();
                }
            }
        }
        return single;
    }
}
