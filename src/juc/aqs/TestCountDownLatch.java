package juc.aqs;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.locks.ReentrantLock;

/**
 * @author xwp
 * @date 2024/5/23
 * @Description
 */
        /*
                awite()时，如果state = 0才会苏醒，和go的awaitGroup有点像
         */
public class TestCountDownLatch {
    public static void main(String[] args) throws InterruptedException {
        CountDownLatch count = new CountDownLatch(10);
        ExecutorService threadPool = Executors.newCachedThreadPool();
        for (int i = 0; i < 10; i++) {
            threadPool.submit(()->{
                try {
                    System.out.println(Thread.currentThread().getName()+" process...");
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                } finally {
                    System.out.println("process finish");
                    count.countDown();
                }
            });
        }
        count.await();
        System.out.println("all finished");
    }
}
