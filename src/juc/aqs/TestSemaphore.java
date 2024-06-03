package juc.aqs;

import java.util.concurrent.*;
import java.util.concurrent.locks.StampedLock;

/**
 * @author xwp
 * @date 2024/5/23
 * @Description
 */

        /*
            信号量10就是多个ReentrantLock
         */
public class TestSemaphore {
    public static void main(String[] args) throws InterruptedException {
        ExecutorService threadPool = Executors.newCachedThreadPool();
        Semaphore semaphore = new Semaphore(3);
        for (int i = 0; i < 10; i++) {
            threadPool.submit(()->{
                try {
                    semaphore.acquire();
                    System.out.println(Thread.currentThread().getName()+" process...");
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                } finally {
                    semaphore.release();
                }
            });
        }
        threadPool.awaitTermination(10, TimeUnit.SECONDS);
        System.out.println("all finished");
    }
}
