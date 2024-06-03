package juc.aqs;

import java.util.concurrent.*;

/**
 * @author xwp
 * @date 2024/5/23
 * @Description
 */

        /*
                基于ReentranLock + Condition 实现工作线程阻塞，只有等到state = 0才会释放
         */
public class TestCyclicBarrier {
    public static void main(String[] args) throws InterruptedException {
        CyclicBarrier count = new CyclicBarrier(10);
        ExecutorService threadPool = Executors.newCachedThreadPool();
        for (int i = 0; i < 10; i++) {
            threadPool.submit(()->{
                try {
                    System.out.println(Thread.currentThread().getName()+" process...");
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                } finally {
                    try {
                        count.await();
                    } catch (Exception e) {
                        e.printStackTrace();
                    } finally {
                        System.out.println("process finish");
                    }
                }
            });
        }
        threadPool.awaitTermination(5,TimeUnit.SECONDS);
        System.out.println("all finished");
        threadPool.shutdown();
    }
}
