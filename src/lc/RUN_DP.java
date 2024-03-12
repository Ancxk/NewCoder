package lc;

import java.util.Arrays;

/**
 * @author xwp
 * @date 2024/3/5
 * @Description 动态规划
 */
public class RUN_DP {
    public static void main(String[] args) {

    }
}

/**
 * 打家劫舍
 * https://leetcode.cn/problems/house-robber/description/
 */
class Solution198 {
    int [] memo =null;
    public int rob(int[] nums) {
//        memo = new int[nums.length];
//        Arrays.fill(memo,-1);
//        return dfs(nums,nums.length-1);

        //dp
//        int len = nums.length;
//        int[] f = new int[len+2];
//        for (int i = 0; i < len; i++) {
//            f[i+2] = Math.max(f[i+1],f[i]+nums[i]);
//        }
//        return f[len+1];

        // 滚动数组
        int len = nums.length;
        int[] f = new int[len+2];
        int f0 = 0;
        int f1 = 0
                ;
        for (int i = 0; i < len; i++) {
            int new_f = Math.max(f1,f0+nums[i]);
            f0 = f1;
            f1 = new_f;
        }
        return f1;


    }

    //选与不选,使用memo数组存储重复计算的值
    public int dfs(int[] nums,int i) {
        if(i < 0) {
            return 0;
        }
        if  (memo[i] != -1){
            return memo[i];
        }
        int ans = Math.max(dfs(nums,i-1),dfs(nums,i-2)+nums[i]);
        memo[i] = ans;
        return ans;
    }


}


/**
 * 打家劫舍2
 * https://leetcode.cn/problems/house-robber-ii/description/
 */
class Solution213 {
    public int rob(int[] nums) {
        int len = nums.length;
        if(len == 1) return nums[0];
        return Math.max(rob2(nums,0,len-2),rob2(nums,1,len-1));
    }
    public int rob2(int[] nums,int i,int j){
        int f0 = 0,f1 = 0;
        for (int k = i; k <= j; k++) {
            int new_f = Math.max(f1,f0+nums[k]);
            f0 = f1;
            f1 = new_f;
        }
        return f1;
    }
}


/**
 * 经典背包问题
 */
class SolutionDp{
    //w物品重量，v物品价值,free空间大小
    int[] w,v;
    public int findMaxValue(int[] w,int[] v,int free){
        this.w = w;
        this.v = v;
        int len = w.length;
        return dfs(len-1,free);

    }
    public int dfs(int i,int free){
        if(i < 0) return 0;
        if(free < w[i]){
            return dfs(i-1,free);
        }
        return Math.max(dfs(i-1,free),dfs(i-1,free-w[i])+v[i]);
    }

}


/**
 * 目标和
 * https://leetcode.cn/problems/target-sum/description/
 */
class Solution494{
    int len,ans;
    int[][] memo;
    public int findTargetSumWays(int[] nums, int target) {
//        len = nums.length;
//        memo = new int[len][target+1];
//        for (int i = 0; i < len; i++) {
//            Arrays.fill(memo[i],-1);
//        }
//        for (int num : nums) {
//            target += num;
//        }
//        if (target < 0 || target%2 == 1) return  0;
//        return dfs2(nums,len-1,target/2);
    }

    //选与不选
    public void dfs(int[] nums,int i,int tar,int sum){
        if(i == len){
            if(tar == sum){
                ans++;
            }
            return;
        }
        dfs(nums,i+1,tar,sum+nums[i]);
        dfs(nums,i+1,tar,sum-nums[i]);
    }
    //背包问题,

    public int dfs2(int[] nums,int i, int free){
        if(i < 0){
            return free == 0 ? 1 : 0;
        }
        if(memo[i][free] != -1){
            return memo[i][free];
        }
        if(nums[i] > free){
            int k = dfs2(nums,i-1,free+nums[i]);
            memo[i][free] = k;
            return k;
        }
        int k= dfs2(nums,i-1,free+nums[i])+dfs2(nums,i-1,free-nums[i]);
        memo[i][free] = k;
        return k;
    }
    //换成地推
    public int dfs3(int[] nums, int target){
        for (int num : nums) {
            target += num;
        }
        len = nums.length;
        if (target < 0 || target%2 == 1) return  0;
        target /=2 ;
        int[][] dp = new int[len+1][target+1];
        for (int i = 0; i < len; i++) {
            for (int j = 0; j <= target ; j++) {
                if(nums[i] > j){
                    dp[i+1][j] = dp[i][j];
                }else{
                    dp[i+1][j] = dp[i][j] + dp[i][j-nums[i]];
                }
            }
        }
        return dp[len][target];
    }


}
