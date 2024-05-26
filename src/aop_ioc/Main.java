package aop_ioc;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;


public class Main {
    public static void main(String[] args) {
        ServiceImpl serviceImpl = new ServiceImpl();
        Service serviceProxy = (Service) Proxy.newProxyInstance(
                serviceImpl.getClass().getClassLoader(),
                serviceImpl.getClass().getInterfaces(),
                new LoggingHandler(serviceImpl)
        );
        serviceProxy.perform();
    }
}


 class LoggingHandler implements InvocationHandler {

    private final Object target;

    public LoggingHandler(Object target) {
        this.target = target;
    }

    @Override
    public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
        System.out.println("Method is about to be called");
        return method.invoke(target, args);
    }
}

 interface Service {
    void perform();
}

 class ServiceImpl implements Service {
    @Override
    public void perform() {
        System.out.println("Service is performing");
    }
}


