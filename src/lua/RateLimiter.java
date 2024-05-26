package lua;

/**
 * @author xwp
 * @date 2024/4/29
 * @Description
 */
public interface RateLimiter {
    boolean Allow(String key);
}
