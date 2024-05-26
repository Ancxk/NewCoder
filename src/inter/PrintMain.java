package inter;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * 交替打印abc
 * @author xwp
 * @date 2024/4/20
 * @Description
 */
public class PrintMain {
    public static void main(String[] args) {
        PrintAB printAB = new PrintAB();
        printAB.print();
    }
}


class PrintAB {
    AtomicInteger i = new AtomicInteger();
    private volatile int p = 0;
    private final Object obj = new Object();

    public void funcPrint(int i,String s){
        while (p < 100) {
            synchronized (obj) {
                while (p % 3 != i) {
                    try {
                        obj.wait();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
                if(p >100) break;
                System.out.println(s+" " + p + Thread.currentThread().getName());
                p++;
                obj.notifyAll();
            }
        }
    }
    public void print() {
        ExecutorService threadPool = Executors.newFixedThreadPool(3);
        threadPool.execute(() -> {
            funcPrint(0,"a");
        });
        threadPool.execute(() -> {
            funcPrint(1,"b");
        });
        threadPool.execute(() -> {
            funcPrint(2,"c");
        });
        threadPool.shutdown();
    }
}


