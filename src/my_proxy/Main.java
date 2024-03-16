package my_proxy;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;

// 抽象类DataSource
abstract class DataSource {
    // 抽象方法getConnection
    public abstract Connection getConnection() throws SQLException;
}

// BasicDataSource类
class BasicDataSource extends DataSource {
    // 实现getConnection方法
    @Override
    public Connection getConnection() throws SQLException {
        // 返回未被代理的Connection对象
        return DriverManager.getConnection("jdbc:mysql://localhost:3306/upmeedb", "root", "123456");
    }
}

// DBCPDataSource类
class DBCPDataSource extends DataSource {
    private ConnectionPool connectionPool;

    public DBCPDataSource(ConnectionPool connectionPool) {
        this.connectionPool = connectionPool;
    }

    // 实现getConnection方法
    @Override
    public Connection getConnection() throws SQLException {
        // 返回通过Proxy代理的Connection对象
        Connection realConnection = DriverManager.getConnection("jdbc:mysql://localhost:3306/upmeedb", "root", "123456");
        ConnectionHandler handler = new ConnectionHandler(realConnection, connectionPool);
        return (Connection) Proxy.newProxyInstance(
                realConnection.getClass().getClassLoader(),
                new Class[]{Connection.class},
                handler
        );
    }
}

// ConnectionHandler类实现InvocationHandler接口
class ConnectionHandler implements InvocationHandler {
    private final Connection realConnection;
    private final ConnectionPool connectionPool;

    public ConnectionHandler(Connection realConnection, ConnectionPool connectionPool) {
        this.realConnection = realConnection;
        this.connectionPool = connectionPool;
    }

    @Override
    public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
        if (method.getName().equals("close")) {
            connectionPool.returnConnection(realConnection); // 拦截close方法，将连接放回连接池
            return null; // 不执行原本的close方法
        } else {
            return method.invoke(realConnection, args); // 其他方法直接执行
        }
    }
}

// ConnectionPool类用于管理数据库连接池
class ConnectionPool {
    // 连接池逻辑，这里只是一个简单示例
    public void returnConnection(Connection connection) {
        // 将连接放回连接池
    }
}

// JDBCTemplate类
class JDBCTemplate {
    private DataSource dataSource;

    // 构造器注入
    public JDBCTemplate(DataSource dataSource) {
        this.dataSource = dataSource;
    }

    // setter方法注入
    public void setDataSource(DataSource dataSource) {
        this.dataSource = dataSource;
    }

    // 使用数据源执行操作的方法
    public void execute() {
        try {
            Connection connection = dataSource.getConnection();
            // 执行数据库操作
            // ...
            connection.close(); // 关闭连接
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
}

public class Main {
    public static void main(String[] args) {
        // 创建连接池
        ConnectionPool connectionPool = new ConnectionPool();

        // 创建数据源
        DataSource basicDataSource = new BasicDataSource();
        DataSource dbcpDataSource = new DBCPDataSource(connectionPool);

        // 使用构造器注入创建JDBCTemplate
        JDBCTemplate jdbcTemplate1 = new JDBCTemplate(basicDataSource);
        jdbcTemplate1.execute();

        // 使用setter方法注入创建JDBCTemplate
        JDBCTemplate jdbcTemplate2 = new JDBCTemplate(dbcpDataSource);
        jdbcTemplate2.execute();
    }
}
