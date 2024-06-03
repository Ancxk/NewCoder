package test;

import java.util.HashMap;
import java.util.Hashtable;
import java.util.concurrent.*;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

/**
 * @author xwp
 * @date 2024/5/23
 * @Description
 */
public class Main {
    public static void main(String[] args) throws InterruptedException {
        ReentrantLock rlock = new ReentrantLock(true);
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


        /*
        sync锁升级原理：当只有第一个线程走到代码块中时，会再对象头中设置程偏向锁，对象头存当前TheadId,下次这个线程获取
        如果又获取到锁，则会比较TheadID，相同则执行逻辑。
        1.升级轻量锁：如果线程B尝试获取锁，发现TheadID不是自己的，，通过cas
        把自己栈帧cp的对象头信息与当前对象头比较，如果相同，则执行，如果不同，则会判断当前对象头里的存的时当前Thead的
        栈帧锁记录，如果相同则锁重入，获取成功。获取失败，则会升级重量锁，
        2.重量锁：会生成一个moniter对象，对象头存的就是这个moniter指针，moniter中有onwer,wait_que,entry_que

         */
    }
}
