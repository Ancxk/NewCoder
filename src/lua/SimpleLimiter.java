package lua;

/**
 * @author xwp
 * @date 2024/4/29
 * @Description
 */
public class SimpleLimiter extends AbstractRateLimiterStrategy{

    @Override
    public boolean Allow(String key) {
        return false;
    }
}
