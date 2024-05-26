package lua;

/**
 * @author xwp
 * @date 2024/4/29
 * @Description
 */
public abstract class AbstractRateLimiterStrategy implements RateLimiter {


    public boolean NeedAllowN(String key,int cost) {
        return false;
    }
}
