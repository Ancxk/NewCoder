package lc;

import java.net.Socket;
import java.util.*;

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

    ListNode() {
    }

    public ListNode(int val) {
        this.val = val;
    }

    ListNode(int val, ListNode next) {
        this.val = val;
        this.next = next;
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
        if (stack2.size() > 0) {
            return stack2.peek();
        } else {
            while (stack1.size() > 0) {
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
        while (que1.size() > 1) {
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
            Arrays.fill(ints, -1);
        }
        return dfs(c, 0, len - 1);
    }

    public int dfs(char[] c, int i, int j) {
        if (i > j) return 0;
        if (i == j) return 1;
        if (memo[i][j] != -1) return memo[i][j];
        if (c[i] == c[j]) return memo[i][j] = dfs(c, i + 1, j - 1) + 2;
        return memo[i][j] = Math.max(dfs(c, i + 1, j), dfs(c, i, j - 1));
    }

    //翻译成递推
    public int longestPalindromeSubseq2(String s) {
        char[] c = s.toCharArray();
        int len = s.length();
        int[][] dp = new int[len][len];
        dp[len - 1][len - 1] = 1;
        for (int i = len - 2; i >= 0; i--) {
            dp[i][i] = 1;
            for (int j = i + 1; j < len; j++) {
                if (c[i] == c[j]) dp[i][j] = dp[i + 1][j - 1] + 2;
                else {
                    dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[0][len - 1];
    }
}

/*
https://leetcode.cn/problems/minimum-score-triangulation-of-polygon/
 */
class Solution1039 {
    //dp
    public int minScoreTriangulation3(int[] values) {
        int len = values.length;
        int[][] dp = new int[len + 1][len + 1];
        for (int i = len - 3; i >= 0; i--) {
            for (int j = i + 1; j < len; j++) {
                if (j - i < 2) continue;
                dp[i][j] = Integer.MAX_VALUE;
                for (int k = i + 1; k < j; k++) {
                    int v = values[i] * values[j] * values[k];
                    dp[i][j] = Math.min(dp[i][j], dp[i][k] + dp[k][j] + v);
                }
            }
        }
        return dp[0][len - 1];
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
        String s = word1 + word2;
        char[] c = s.toCharArray();
        int len = c.length;
        int ans = 0;
        int[][] dp = new int[len][len];
        dp[len - 1][len - 1] = 1;
        for (int i = len - 2; i >= 0; i--) {
            dp[i][i] = 1;
            for (int j = i + 1; j < len; j++) {
                if (c[i] == c[j]) {
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                    if (i < word1.length() && j >= word1.length()) {
                        ans = Math.max(ans, dp[i][j]);
                    }
                } else {
                    dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]);
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

/*
方案数鼻祖-爬楼梯
https://leetcode.cn/problems/climbing-stairs/
 */
class Solution70 {
    public int climbStairs(int n) {
        int[] dp = new int[n + 1];
        if (n == 1) return 1;
        dp[0] = 1;
        dp[1] = 2;
        for (int i = 0; i < n; i++) {
            dp[i + 2] = dp[i + 1] + dp[i];
        }
        return dp[n];
    }
}

/*
https://leetcode.cn/problems/min-cost-climbing-stairs/
 */
class Solution746 {
    public int minCostClimbingStairs(int[] cost) {
        int n = cost.length;
        int[] dp = new int[n + 1];
        dp[0] = 0;
        dp[1] = 0;
        for (int i = 2; i <= n; i++) {
            dp[i] = Math.min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2]);
        }
        return dp[n];
    }
}

/*
组合总和4,
爬楼梯变种，只不过每次可以爬n次，n从数组中取！,也相当于零钱兑换！
https://leetcode.cn/problems/combination-sum-iv/description/
 */
class Solution377 {
    public int combinationSum4(int[] nums, int tar) {
        int[] dp = new int[tar + 1];
        final int MOD = 1_000_000_007;
        dp[0] = 1;
        for (int i = 1; i <= tar; i++) {
            for (int num : nums) {
                if (i - num >= 0) dp[i] += dp[i - num];
            }
        }
        return dp[tar];
    }
}

/*
https://leetcode.cn/problems/count-ways-to-build-good-strings/
爬楼梯变种！
 */
class Solution2466 {
    public int countGoodStrings(int low, int high, int zero, int one) {
        int[] dp = new int[high + 1];
        dp[0] = 1;
        int res = 0;
        final int MOD = 1_000_000_007;
        for (int i = 1; i <= high; i++) {
            int a = i - zero < 0 ? 0 : dp[i - zero];
            dp[i] = (a + dp[i]) % MOD;
            int b = i - one < 0 ? 0 : dp[i - one];
            dp[i] = (b + dp[i]) % MOD;
            if (i >= low) {
                res = (res + dp[i]) % MOD;
            }
        }
        return res;
    }
}

/*
01背包
背包问题，选与不选，for外层枚举物品，内层枚举空间，与爬楼梯不是一个类型的。
 */
class Solution494_2 {
    public int findTargetSumWays(int[] nums, int target) {
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        int k = target + sum;
        if (k < 0 || k % 2 == 1) return 0;
        k /= 2;
        //背包大小为k，枚举物品
        int n = nums.length;
        int[][] f = new int[n + 1][k + 1];
        f[0][0] = 1;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= k; j++) {
                if (nums[i] > j) f[i + 1][j] = f[i][j];
                else f[i + 1][j] = f[i][j - nums[i]] + f[i][j];
            }
        }
        return f[n][k];

    }
}


/*完全背包
https://leetcode.cn/problems/coin-change/
 */
class Solution322_2 {
    static final int MAX = Integer.MAX_VALUE / 2;

    public int coinChange(int[] coins, int amount) {
        int n = coins.length;
        int[][] dp = new int[n + 1][amount + 1];
        for (int[] ints : dp) {
            Arrays.fill(ints, MAX);
        }
        dp[0][0] = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= amount; j++) {
                if (j - coins[i] < 0) {
                    dp[i + 1][j] = dp[i][j];
                } else {
                    dp[i + 1][j] = Math.min(dp[i][j], dp[i + 1][j - coins[i]] + 1);
                }
            }
        }
        return dp[n][amount] == MAX ? -1 : dp[n][amount];
    }
}


class Solution494_3 {
    public int findTargetSumWays(int[] nums, int target) {
        int sum = 0;
        for (int i : nums) {
            sum += i;
        }
        int n = nums.length;
        int x = sum + target;
        if (x < 0 || x % 2 == 1) return 0;
        x /= 2;
        int[][] dp = new int[n + 1][x + 1];
        dp[0][0] = 1;
        //dp[i][j]的含义就是从前i个物品中，凑齐j的方法个数
        for (int i = 0; i < n; i++) {
            for (int j = x; j >= 0; j--) {
                if (j < nums[i]) {
                    dp[i + 1][j] = dp[i][j];
                } else {
                    dp[i + 1][j] = dp[i][j] + dp[i][j - nums[i]];
                }
            }
        }

        int[] f = new int[x + 1];
        for (int i = 0; i < n; i++) {
            for (int j = x; j >= 0; j--) {
                f[j] += f[j - nums[i]];
            }
        }
        //return f[x];
        return dp[n][x];
    }

    public int dfs(int[] nums, int i, int c) {
        if (i < 0) return 0;
        if (i == 0) return 1;
        if (c < nums[i]) return dfs(nums, i - 1, c);
        return dfs(nums, i - 1, c) + dfs(nums, i, c - nums[i]);
    }
}

class Solution322_3 {
    public int coinChange(int[] coins, int amount) {
        int n = coins.length;
        int[][] f = new int[n + 1][amount + 1];
        for (int i = 0; i <= n; i++) {
            Arrays.fill(f[i], Integer.MAX_VALUE / 2);
        }
        //f[i][j]表示以i结尾的数组，剩余背包为i的使用硬币最小个数。
        f[0][0] = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= amount; j++) {
                if (coins[i] > j) {
                    f[i + 1][j] = f[i][j];
                } else {
                    f[i + 1][j] = Math.min(f[i + 1][j - coins[i]] + 1, f[i][j]);
                }
            }
        }
        int i = f[n][amount];
        return i == Integer.MAX_VALUE / 2 ? -1 : i;
    }

    public int coinChange2(int[] coins, int amount) {
        int[] ff = new int[amount + 1];
        Arrays.fill(ff, Integer.MAX_VALUE);
        ff[0] = 0;
        for (int coin : coins) {
            for (int j = 0; j <= amount; j++) {
                if (coin <= j) {
                    ff[j] = Math.min(ff[j - coin] + 1, ff[j]);
                }
            }
        }
        return ff[amount] == Integer.MAX_VALUE ? -1 : ff[amount];
    }
}


/*
https://leetcode.cn/problems/length-of-the-longest-subsequence-that-sums-to-target/
 零钱兑换翻版，一个物品只能选一次，并且返回最大物品个数。
 */
class Solution2915_2 {
    public int lengthOfLongestSubsequence(List<Integer> nums, int tar) {
        int n = nums.size();
        int[][] f = new int[n + 1][tar + 1];
        for (int[] ints : f) {
            Arrays.fill(ints, Integer.MIN_VALUE);
        }
        f[0][0] = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= tar; j++) {
                if (j - nums.get(i) >= 0) {
                    f[i + 1][j] = Math.max(f[i][j], f[i][j - nums.get(i)] + 1);
                } else {
                    f[i + 1][j] = f[i][j];
                }
            }
        }
        int x = f[n][tar];
        return x < 0 ? -1 : x;
    }

    //滚动数组
    public int lengthOfLongestSubsequence2(List<Integer> nums, int tar) {
        int[] f = new int[tar + 1];
        Arrays.fill(f, Integer.MIN_VALUE);
        f[0] = 0;
        for (Integer num : nums) {
            for (int j = tar; j >= 0; j--) {
                if (j - num >= 0) {
                    f[j] = Math.max(f[j], f[j - num] + 1);
                }
            }
        }
        int x = f[tar];
        return x < 0 ? -1 : x;
    }
}


/*
https://leetcode.cn/problems/partition-equal-subset-sum/
 */
class Solution416 {
    public boolean canPartition(int[] nums) {
        int sum = 0;
        for (int i : nums) {
            sum += i;
        }
        if (sum % 2 == 1) return false;
        sum /= 2;
        var n = nums.length;
        int[][] f = new int[n + 1][sum + 1];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= sum; j++) {
                if (j - nums[i] >= 0 && f[i][j - nums[i]] + nums[i] == j) {
                    f[i + 1][j] = j;
                } else {
                    f[i + 1][j] = f[i][j];
                }
            }
        }
        return f[n][sum] == sum;
    }

    //滚动数组
    public boolean canPartition2(int[] nums) {
        int sum = 0;
        for (int i : nums) {
            sum += i;
        }
        if (sum % 2 == 1) return false;
        sum /= 2;
        var n = nums.length;
        int[] f = new int[sum + 1];
        for (int num : nums) {
            for (int j = sum; j >= 0; j--) {
                if (j - num >= 0 && f[j - num] + num == j) {
                    f[j] = j;
                }
            }
        }
        return f[sum] == sum;
    }

}


/*
https://leetcode.cn/problems/coin-change-ii/
零钱兑换2，递归没懂。。。拆分子问题不懂为啥这么拆
 */
class Solution518_2 {
    public int change(int amount, int[] coins) {
        int n = coins.length;
        int[][] f = new int[n + 1][amount + 1];
        f[0][0] = 1;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= amount; j++) {
                if (j - coins[i] >= 0) {
                    f[i + 1][j] = f[i + 1][j - coins[i]] + f[i][j];
                } else {
                    f[i + 1][j] = f[i][j];
                }
            }
        }
        return f[n][amount];
    }

    public int dfs(int[] nums, int i, int c) {
        if (i < 0) {
            return c == 0 ? 1 : 0;
        }
        if (c - nums[i] < 0) return dfs(nums, i - 1, c);
        return dfs(nums, i - 1, c) + dfs(nums, i, c - nums[i]);
    }
}

/*
https://leetcode.cn/problems/longest-common-subsequence/
 */

class Solution1143_2 {
    int[][] memo;

    public int longestCommonSubsequence(String text1, String text2) {
        int a = text1.length(), b = text2.length();
        memo = new int[a][b];
        for (int[] ints : memo) {
            Arrays.fill(ints, -1);
        }
        return dfs(text1, text2, text1.length() - 1, text2.length() - 1);

    }

    public int dfs(String t1, String t2, int i, int j) {
        if (i < 0 || j < 0) return 0;
        if (memo[i][j] != -1) return memo[i][j];
        if (t1.charAt(i) == t2.charAt(j)) return memo[i][j] = dfs(t1, t2, i - 1, j - 1) + 1;
        return memo[i][j] = Math.max(dfs(t1, t2, i - 1, j), dfs(t1, t2, i, j - 1));
    }
}

/*
完全背包，最小个数
 */
class Solution322_4 {
    int M = Integer.MAX_VALUE / 2;

    public int coinChange(int[] coins, int amount) {
        int n = coins.length;
        int[][] f = new int[n + 1][amount + 1];
        int k = dfs(coins, n - 1, amount);
        return k == M ? -1 : k;

    }

    public int coinChange3(int[] coins, int amount) {
        int n = coins.length;
        int[] f = new int[amount + 1];
        Arrays.fill(f, Integer.MAX_VALUE / 2);
        f[0] = 0;
        for (int coin : coins) {
            for (int j = 0; j <= amount; j++) {
                if (coin <= j) {
                    f[j] = Math.min(f[j - coin] + 1, f[j]);
                }
            }
        }
        return f[amount] == Integer.MAX_VALUE / 2 ? -1 : f[amount];

    }

    public int coinChange2(int[] coins, int amount) {
        int n = coins.length;
        int[][] f = new int[n + 1][amount + 1];
        for (int[] ints : f) {
            Arrays.fill(ints, Integer.MAX_VALUE / 2);
        }
        f[0][0] = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= amount; j++) {
                if (coins[i] > j) {
                    f[i + 1][j] = f[i][j];
                } else {
                    f[i + 1][j] = Math.min(f[i + 1][j - coins[i]] + 1, f[i][j]);
                }
            }
        }
        return f[n][amount] == Integer.MAX_VALUE / 2 ? -1 : f[n][amount];
    }

    public int dfs(int[] coin, int i, int c) {
        if (i < 0) {
            return c == 0 ? 0 : M;
        }
        if (coin[i] > c) return dfs(coin, i - 1, c);
        int d = Math.min(dfs(coin, i - 1, c), dfs(coin, i, c - coin[i]) + 1);
        return d;
    }
}

/*
组合问题
 */
class Solution416_2 {
    int[][] memo;
    public boolean canPartition(int[] nums) {
        int n = nums.length, sum = 0;
        for (int i : nums) {
            sum += i;
        }
        if (sum % 2 == 1) return false;
        sum /= 2;
        memo = new int[n][sum + 1];
        for (int[] ints : memo) {
            Arrays.fill(ints, -1);
        }
        return dfs(nums, n - 1, sum);
    }

    public boolean dfs(int[] n, int i, int c) {
        if (i < 0) return c == 0;
        if (memo[i][c] != -1) return memo[i][c] == 1;
        boolean k = dfs(n, i - 1, c) || c >= n[i] && dfs(n, i - 1, c - n[i]);
        memo[i][c] = k ? 1 : 0;
        return k;
    }
}

/*
完全背包，组合数问题

 */
class Solution518_3 {
    public int change(int amount, int[] coins) {
        int n = coins.length;
        int[][] f = new int[n+1][amount+1];
        for (int i = 0; i <= n; i++) {
            f[i][0] = 1;
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= amount; j++) {
                if(j>=coins[i]) f[i+1][j] = f[i+1][j-coins[i]]+f[i][j];
                else f[i+1][j] = f[i][j];
            }
        }
        return f[n][amount];
    }
    public int change2(int amount, int[] coins) {
        int n = coins.length;
        int[] f = new int[amount+1];
        f[0] = 1;
        for (int coin : coins) {
            for (int j = 0; j <= amount; j++) {
                if (j >= coin) f[j] += f[j - coin];
            }
        }
        return f[amount];
    }
}

/*
子序列问题
 */
class Solution72_2 {
    public int minDistance(String word1, String word2) {
        int a = word1.length(),b= word2.length();
        int[][] f= new int[a+1][b+1];
        for (int i = 0; i <= a; i++) {
            f[i][0] = i+1;
        }

    return 0;
    }

    public int dfs(String w1, String w2, int i, int j) {
        if(i < 0) return j;
        if(j < 0) return i;
        char a = w1.charAt(i);
        char b= w2.charAt(j);
         dfs(w1,w2,i-1,j-1);

        int k = dfs(w1,w2,i-1,j-1);
        if(a == b) return k;
        var p = dfs(w1,w2,i,j-1);
        var q = dfs(w1,w2,i-1,j);
        return Math.min(q,p);

    }
}
