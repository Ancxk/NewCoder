package lc;

import pass_leecode.Solution;

import java.awt.image.AreaAveragingScaleFilter;
import java.net.Socket;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.ToIntFunction;
import java.util.stream.Collectors;

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
    int[] memo = null;

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
        int[] f = new int[len + 2];
        int f0 = 0;
        int f1 = 0;
        for (int i = 0; i < len; i++) {
            int new_f = Math.max(f1, f0 + nums[i]);
            f0 = f1;
            f1 = new_f;
        }
        return f1;


    }

    //选与不选,使用memo数组存储重复计算的值
    public int dfs(int[] nums, int i) {
        if (i < 0) {
            return 0;
        }
        if (memo[i] != -1) {
            return memo[i];
        }
        int ans = Math.max(dfs(nums, i - 1), dfs(nums, i - 2) + nums[i]);
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
        if (len == 1) {
            return nums[0];
        }
        return Math.max(rob2(nums, 0, len - 2), rob2(nums, 1, len - 1));
    }

    public int rob2(int[] nums, int i, int j) {
        int f0 = 0, f1 = 0;
        for (int k = i; k <= j; k++) {
            int new_f = Math.max(f1, f0 + nums[k]);
            f0 = f1;
            f1 = new_f;
        }
        return f1;
    }
}


/**
 * 经典背包问题
 */
class SolutionDp {
    //w物品重量，v物品价值,free空间大小
    int[] w, v;

    public int findMaxValue(int[] w, int[] v, int free) {
        this.w = w;
        this.v = v;
        int len = w.length;
        return dfs(len - 1, free);

    }

    public int dfs(int i, int free) {
        if (i < 0) {
            return 0;
        }
        if (free < w[i]) {
            return dfs(i - 1, free);
        }
        return Math.max(dfs(i - 1, free), dfs(i - 1, free - w[i]) + v[i]);
    }

}


/**
 * 目标和
 * https://leetcode.cn/problems/target-sum/description/
 */
class Solution494 {
    int len, ans;
    int[][] memo;

    public int findTargetSumWays(int[] nums, int target) {
        return dfs4(nums, target);
    }

    //选与不选
    public void dfs(int[] nums, int i, int tar, int sum) {
        if (i == len) {
            if (tar == sum) {
                ans++;
            }
            return;
        }
        dfs(nums, i + 1, tar, sum + nums[i]);
        dfs(nums, i + 1, tar, sum - nums[i]);
    }
    //背包问题,

    public int dfs2(int[] nums, int i, int free) {
        if (i < 0) {
            return free == 0 ? 1 : 0;
        }
        if (memo[i][free] != -1) {
            return memo[i][free];
        }
        if (nums[i] > free) {
            int k = dfs2(nums, i - 1, free + nums[i]);
            memo[i][free] = k;
            return k;
        }
        int k = dfs2(nums, i - 1, free + nums[i]) + dfs2(nums, i - 1, free - nums[i]);
        memo[i][free] = k;
        return k;
    }

    //换成地推
    public int dfs3(int[] nums, int target) {
        for (int num : nums) {
            target += num;
        }
        len = nums.length;
        if (target < 0 || target % 2 == 1) {
            return 0;
        }
        target /= 2;
        int[][] dp = new int[len + 1][target + 1];
        for (int i = 0; i < len; i++) {
            for (int j = 0; j <= target; j++) {
                if (nums[i] > j) {
                    dp[i + 1][j] = dp[i][j];
                } else {
                    dp[i + 1][j] = dp[i][j] + dp[i][j - nums[i]];
                }
            }
        }
        return dp[len][target];
    }

    //滚动数组。。。
    public int dfs4(int[] nums, int target) {
        for (int num : nums) {
            target += num;
        }
        len = nums.length;
        if (target < 0 || target % 2 == 1) {
            return 0;
        }
        target /= 2;
        int[] dp = new int[target + 1];
        for (int i = 0; i < len; i++) {
            //为啥要从tar开始遍历？
            //如果从j = 0开始遍历，for——j每次从已经改变的dp[j]中获取，
            //也就是说第二层遍历会复盖之前的
            for (int j = target; j >= 0; j--) {
                if (nums[i] <= j) {
                    dp[j] = dp[j] + dp[j - nums[i]];
                }
            }
        }
        return dp[target];
    }

}

/**
 * 零钱兑换
 * https://leetcode.cn/problems/coin-change/description/
 */
class Solution322 {
    public int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, Integer.MAX_VALUE / 2);
        dp[0] = 0;
        for (int coin : coins) {
            for (int j = 1; j <= amount; j++) {
                if (coin <= j) {
                    dp[j] = Math.min(dp[j], dp[j - coin] + 1);
                }
            }
        }
        return dp[amount] == Integer.MAX_VALUE / 2 ? -1 : dp[amount];
    }

    //完全背包
    public int coinChange2(int[] coins, int amount) {
        int len = coins.length;
        int[][] dp = new int[len + 1][amount + 1];
        for (int[] ints : dp) {
            Arrays.fill(ints, Integer.MAX_VALUE / 2);
        }
        dp[0][0] = 0;
        for (int i = 0; i < len; i++) {
            for (int j = 0; j <= amount; j++) {
                if (coins[i] > j) {
                    dp[i + 1][j] = dp[i][j];
                } else {
                    dp[i + 1][j] = Math.min(dp[i][j], dp[i + 1][j - coins[i]] + 1);
                }
            }
        }
        return dp[len][amount] == Integer.MAX_VALUE / 2 ? -1 : dp[len][amount];
    }

    //递归
    public int coinChange3(int[] coins, int amount) {
        int i = dfs(coins, coins.length - 1, amount);
        return i == Integer.MAX_VALUE / 2 ? -1 : i;
    }

    public int dfs(int[] coins, int i, int c) {
        if (i < 0) {
            return c == 0 ? 1 : Integer.MAX_VALUE / 2;
        }
        if (coins[i] > c) {
            return dfs(coins, i - 1, c);
        }
        //还是从i开始选
        return Math.min(dfs(coins, i - 1, c), dfs(coins, i, c - coins[i]) + 1);
    }


}

/**
 * https://leetcode.cn/problems/length-of-the-longest-subsequence-that-sums-to-target/
 */
class Solution2915 {
    //妈的，自己写出来了，但不懂。。。。
    public int lengthOfLongestSubsequence(List<Integer> nums, int target) {
        int[] dp = new int[target + 1];
        Arrays.fill(dp, -1);
        dp[0] = 0;
        for (Integer num : nums) {
            for (int j = target; j >= 0; j--) {
                if (num <= j && dp[j - num] != -1) {
                    dp[j] = Math.max(dp[j], dp[j - num] + 1);
                }
            }
        }
        return dp[target];
    }

    public int lengthOfLongestSubsequence2(List<Integer> nums, int target) {
        int k = dfs(nums, nums.size() - 1, target);
        return k < 0 ? -1 : k;
    }

    public int dfs(List<Integer> nums, int i, int c) {
        if (i < 0) {
            return c == 0 ? 0 : Integer.MIN_VALUE / 2;
        }
        if (nums.get(i) >= c) {
            return dfs(nums, i - 1, c);
        }
        return Math.max(dfs(nums, i - 1, c), dfs(nums, i - 1, c - nums.get(i)) + 1);
    }


    public int lengthOfLongestSubsequence3(List<Integer> nums, int target) {
        int len = nums.size();
        int[] dp = new int[target + 1];
        Arrays.fill(dp, Integer.MIN_VALUE);
        dp[0] = 0;
        for (int i = 0; i < len; i++) {
            for (int j = target; j >= 0; j--) {
                if (j >= nums.get(i))
                    dp[j] = Math.max(dp[j - nums.get(i)] + 1, dp[j]);
            }
        }
        return dp[target];
    }
}


/**
 * https://leetcode.cn/problems/longest-common-subsequence/
 */
class Solution1143 {
    public int longestCommonSubsequence(String text1, String text2) {
        char[] c1 = text1.toCharArray();
        char[] c2 = text2.toCharArray();
        int[][] dp = new int[c1.length + 1][c2.length + 1];
        for (int i = 0; i < c1.length; i++) {
            for (int j = 0; j < c2.length; j++) {
                if (c1[i] == c2[j]) {
                    dp[i + 1][j + 1] = dp[i][j] + 1;
                } else {
                    dp[i + 1][j + 1] = Math.max(dp[i][j + 1], dp[i + 1][j]);
                }
            }
        }
        return dp[c1.length][c2.length];
    }

    public int longestCommonSubsequence2(String text1, String text2) {
        char[] c1 = text1.toCharArray();
        char[] c2 = text2.toCharArray();
        return dfs(c1, c2, c1.length - 1, c2.length - 1);
    }

    public int dfs(char[] c1, char[] c2, int i, int j) {
        if (i < 0 || j < 0) {
            return 0;
        }
        if (c1[i] != c2[j]) {
            return Math.max(dfs(c1, c2, i - 1, j), dfs(c1, c2, i, j - 1));
        }
        return dfs(c1, c2, i - 1, j - 1) + 1;
    }
}

/**
 * https://leetcode.cn/problems/edit-distance/description/
 */
class Solution72 {
    public int minDistance(String word1, String word2) {
        // "horse",
        // "ros"
        var c1 = word1.toCharArray();
        var c2 = word2.toCharArray();
        var len1 = c1.length;
        var len2 = c2.length;
        int[][] dp = new int[len1 + 1][len2 + 1];
        for (int i = 0; i < len1; i++) {
            dp[i][0] = i;
        }
        for (int i = 0; i < len2; i++) {
            dp[0][i] = i;
        }
        for (int i = 0; i < len1; i++) {
            for (int j = 0; j < len2; j++) {
                if (c1[i] == c2[j]) {
                    dp[i + 1][j + 1] = dp[i][j];
                } else {
                    dp[i + 1][j + 1] = Math.min(dp[i][j], Math.min(dp[i][j + 1], dp[i + 1][j])) + 1;
                }
            }
        }
        return dp[len1][len2];
    }
}


/**
 * https://leetcode.cn/problems/longest-increasing-subsequence/
 */
class Solution300 {
    public int lengthOfLIS(int[] nums) {
        int len = nums.length, ans = 1;
        int[] dp = new int[len + 1];
        Arrays.fill(dp, 1);
        for (int i = 0; i < len; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                    ans = Math.max(ans, dp[i]);
                }
            }
        }
        return ans;
    }

    //贪心？
    public int lengthOfLIS2(int[] nums) {
        int len = nums.length, ans = 1;
        return 0;
    }
}

/**
 * https://leetcode.cn/problems/find-peak-element/description/
 * https://leetcode.cn/problems/peak-index-in-a-mountain-array/
 */
class Solution_162_852 {
    public int findPeakElement(int[] nums) {
        int i = 0, j = nums.length - 1;
        while (i <= j) {
            int mid = (i + j) / 2;
            if (nums[mid] < nums[mid + 1]) {
                i = mid + 1;
            } else {
                j = mid - 1;
            }
        }
        return i;
    }

    int len;

    public int peakIndexInMountainArray(int[] arr) {
        int i = 0, len = arr.length;
        int j = len - 1;
        while (i <= j) {
            int mid = (i + j) / 2;
            if (arr[mid] < arr[mid + 1]) {
                i = mid + 1;
            } else {
                j = mid - 1;
            }
        }
        return i;
    }
}


/**
 * 难
 * https://leetcode.cn/problems/minimum-number-of-removals-to-make-mountain-array/
 */
class Solution1671 {

    public int minimumMountainRemovals(int[] nums) {
        int len = nums.length;
        return 0;
    }
}

class ListNode {
    public int val;
    public ListNode next;

    public ListNode(int val) {
        this.val = val;
    }
}


/**
 * https://twitter.com/wu_haobin/status/1771138803459293195
 */
class SolutionSimple {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        ListNode dum = new ListNode(-1), d = dum;
        int[] t = new int[]{};
        while (scanner.hasNext("x"))
            for (int i : t) {
                ListNode listNode = new ListNode(i);
                d.next = listNode;
                d = listNode;
            }
        ListNode mid = getMid(dum.next);
        while (mid != null) {
            System.out.println(mid.val);
            mid = mid.next;
        }
    }

    public static ListNode getMid(ListNode node) {
        ListNode h = node;
        if (h == null) return null;
        if (h.next == null) return h;
        ListNode monoHead = node.next;
        while (h != null && h.next != null) {
            ListNode next = h.next;
            h.next = h.next.next;
            h = next;
        }
        ListNode r = reverseNode(monoHead);
        ListNode l = node;
        ListNode dummy = new ListNode(-1), now = dummy;
        while (l != null && r != null) {
            int val = Math.min(l.val, r.val);
            ListNode listNode = new ListNode(val);
            now.next = listNode;
            now = listNode;
            if (val == l.val) l = l.next;
            else {
                r = r.next;
            }
        }
        if (l != null) {
            now.next = l;
        }
        if (r != null) {
            now.next = r;
        }
        return dummy.next;
    }

    public static ListNode reverseNode(ListNode node) {
        ListNode h = node, pre = null;
        while (h != null) {
            ListNode next = h.next;
            h.next = pre;
            pre = h;
            h = next;
        }
        return pre;
    }
}


/**
 * https://leetcode.cn/problems/best-team-with-no-conflicts/description/
 */
class Solution1626 {
    public static void main(String[] args) {
        int[] scores = new int[]{5, 5, 5, 5};
        int[] ages = new int[]{1, 1, 2, 2};
        int i = bestTeamScore(scores, ages);
        System.out.println(i);
    }

    public static int bestTeamScore(int[] scores, int[] ages) {
        int len = ages.length;
        int[][] players = new int[len][2];
        for (int i = 0; i < len; i++) {
            players[i][0] = ages[i];
            players[i][1] = scores[i];
        }
        Arrays.sort(players, (o1, o2) -> o1[0] != o2[0] ? o1[0] - o2[0] : o1[1] - o2[1]);
        int ans = 0;
        int[] dp = new int[len + 1];
        for (int i = 0; i < len; i++) {
            dp[i] = players[i][1];
        }
        for (int i = 0; i < len; i++) {
            int k = dp[i];
            for (int j = 0; j < i; j++) {
                if (players[i][1] >= players[j][1]) {
                    dp[i] = Math.max(dp[i], k + dp[j]);
                }
            }
            ans = Math.max(ans, dp[i]);
        }
        return ans;
    }
}


/*
https://leetcode.cn/problems/coin-change-ii/description/?envType=daily-question&envId=Invalid%20Date
 */
class Solution518 {
    public int change(int amount, int[] coins) {
        int len = coins.length;
        int[] dp = new int[amount + 1];
        dp[0] = 1;
        for (int i = 0; i < len; i++) {
            for (int j = 1; j <= amount; j++) {
                if (j >= coins[i]) {
                    dp[j] += dp[j - coins[i]];
                }
            }
        }
        return dp[amount];
    }
}

/*
https://leetcode.cn/problems/longest-increasing-subsequence-ii/
 */
class Solution2407 {
    public int lengthOfLIS(int[] nums, int k) {
        return 0;
    }
}


/*
https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-with-cooldown/
 */
class Solution309 {
    public int maxProfit(int[] prices) {
        int len = prices.length;
        if (len == 1) return 0;
        int[][] dp = new int[len + 2][2];
        dp[0][1] = -prices[0];
        for (int i = 1; i < len; i++) {
            //当天不持有股票的最大利润-> 昨天持有股票再卖出去，或者昨天不持有股票
            //当天持有股票的最大利润-> 昨天持有的，或者今天要买股票->只能从前天的未持有股票状态买今天的股票
            dp[i + 1][0] = Math.max(dp[i][0], dp[i][1] + prices[i]);
            dp[i + 1][1] = Math.max(dp[i][1], dp[i - 1][0] - prices[i]);
        }
        return dp[len][0];
    }
}

class MySocket {
    public static void main(String[] args) {
        Socket socket = new Socket();

    }

}


/*
https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-iv/description/
 */
// TODO: 2024/4/1
class Solution188 {
    public int maxProfit(int k, int[] prices) {
        return 0;
    }
}


/*
https://leetcode.cn/problems/implement-queue-using-stacks/
 */

/**
 * Your MyQueue object will be   instantiated and called as such:
 * MyQueue obj = new MyQueue();
 * obj.push(x);
 * int param_2 = obj.pop();
 * int param_3 = obj.peek();
 * boolean param_4 = obj.empty();
 */

class MyQueue {
    private Stack<Integer> stack1 = null;
    private Stack<Integer> stack2 = null;
    public MyQueue() {
        stack1 = new Stack<>();
        stack2 = new Stack<>();
    }

    public void push(int x) {
        stack1.add(x);
    }

    public int pop() {
        peek();
        return stack2.pop();
    }

    public int peek() {
        if(stack2.size() > 0){
            return stack2.peek();
        }else{
            while (stack1.size() > 0){
                stack2.add(stack1.pop());
            }
            return stack2.peek();
        }

    }

    public boolean empty() {
        return stack2.size() + stack1.size() == 0;
    }
}


/*
https://leetcode.cn/problems/implement-stack-using-queues/
 */
/**
 * Your MyStack object will be instantiated and called as such:
 * MyStack obj = new MyStack();
 * obj.push(x);
 * int param_2 = obj.pop();
 * int param_3 = obj.top();
 * boolean param_4 = obj.empty();
 */

class MyStack {
    private Deque<Integer> que1 = null;
    private Deque<Integer> que2 = null;

    public MyStack() {
        que1 = new LinkedList<>();
        que2 = new LinkedList<>();
    }

    public void push(int x) {
        que1.add(x);
    }

    public int pop() {
        while (que1.size() > 1){
            que2.add(que1.pop());
        }
        int k = que1.pop();
        Deque<Integer> tmp = que2;
        que2 = que1;
        que1 = tmp;
        return k;
    }

    public int top() {
        int k = pop();
        push(k);
        return k;
    }

    public boolean empty() {
        return que1.size() == 0;
    }
}


/*
https://leetcode.cn/problems/longest-palindromic-subsequence/
 */
class Solution516 {
    //dfs
    int[][] memo = null;
    public int longestPalindromeSubseq(String s) {
        char[] c = s.toCharArray();

        int len = c.length;
        memo = new int[len][len];
        for (int[] ints : memo) {
            Arrays.fill(ints,-1);
        }
        return dfs(c,0,len-1);
    }
    public int dfs (char[] c,int i,int j){
        if(i>j) return 0;
        if(i == j) return 1;
        if(memo[i][j] != -1) return memo[i][j];
        if(c[i] == c[j]) return memo[i][j] = dfs(c,i+1,j-1)+2;
        return memo[i][j] =  Math.max(dfs(c,i+1,j),dfs(c,i,j-1));
    }

    //翻译成递推
        public int longestPalindromeSubseq2(String s) {
            char[] c = s.toCharArray();
            int len = s.length();
            int[][] dp = new int[len][len];
            dp[len-1][len-1] = 1;
            for(int i = len-2;i >= 0; i--){
                dp[i][i] = 1;
                for(int j = i+1; j < len; j++){
                    if(c[i] == c[j]) dp[i][j] = dp[i+1][j-1]+2;
                    else{
                        dp[i][j] = Math.max(dp[i+1][j],dp[i][j-1]);
                    }
                }
            }
            return dp[0][len-1];
       }
}

/*
https://leetcode.cn/problems/minimum-score-triangulation-of-polygon/
 */
class Solution1039 {
    //dp
    public int minScoreTriangulation3(int[] values) {
        int len = values.length;
        int[][] dp = new int[len+1][len+1];
        for (int i = len-3; i >= 0; i--) {
            for (int j = i+1; j < len; j++) {
                if(j-i < 2) continue;
                dp[i][j] = Integer.MAX_VALUE;
                for (int k = i+1; k < j; k++) {
                    int v = values[i]*values[j]*values[k];
                    dp[i][j] = Math.min(dp[i][j],dp[i][k]+dp[k][j]+v);
                }
            }
        }
        return dp[0][len-1];
    }


    //正确dfs
    int[][] memo = null;
    public int minScoreTriangulation2(int[] values) {
        int len = values.length;
        memo = new int[len][len];
        for (int[] ints : memo) {
            Arrays.fill(ints, -1);
        }
        return dfs(values, 0, len - 1);
    }
    public int dfs(int[] v, int i, int j) {
        if (j - i < 2) return 0;
        if (memo[i][j] != -1) return memo[i][j];
        int mi = Integer.MAX_VALUE;
        for (int k = i + 1; k < j; k++) {
            int s = dfs(v, i, k) + dfs(v, k, j) + v[i] * v[j] * v[k];
            mi = Math.min(mi, s);
        }
        return memo[i][j] = mi;
    }

    //超时dfs
    public int minScoreTriangulation(int[] values) {
        return dfs(values);
    }

    public int dfs(int[] v) {
        int l = v.length;
        if (l < 3) return -1;
        if (l == 3) {
            return v[1] * v[0] * v[2];
        }
        int min = Integer.MAX_VALUE;
        for (int i = 0; i < l; i++) {
            int r = v[i] * v[(i + 1) % l] * v[(i - 1) == -1 ? l - 1 : i - 1];
            int[] ints = new int[l - 1];
            for (int j = 0, p = 0; j < l; j++) {
                if (j != i) ints[p++] = v[j];
            }
            int k = dfs(ints);
            min = Math.min(min, k + r);
        }
        return min;
    }

}

/*
https://leetcode.cn/problems/maximize-palindrome-length-from-subsequences/description/
 */
class Solution1771 {
    public int longestPalindrome(String word1, String word2) {
        String s = word1+word2;
        char[] c = s.toCharArray();
        int len = c.length;
        int ans = 0;
        int[][] dp = new int[len][len];
        dp[len-1][len-1] = 1;
        for(int i = len-2; i >= 0; i--){
            dp[i][i] = 1;
            for(int j = i+1; j < len; j++){
                if(c[i] == c[j]){
                    dp[i][j] = dp[i+1][j-1]+2;
                    if(i < word1.length() && j >= word1.length()){
                        ans = Math.max(ans,dp[i][j]);
                    }
                }else{
                    dp[i][j] = Math.max(dp[i+1][j],dp[i][j-1]);
                }
            }
        }
        return ans;
    }
}


/*
https://leetcode.cn/problems/minimum-cost-to-merge-stones/description/
 */
class Solution1000 {
    public int mergeStones(int[] stones, int k) {
        return 0;
    }
}






