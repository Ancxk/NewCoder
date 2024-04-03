package inter;

import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        List<Double> res = new ArrayList<>();
        res.add(0.55);
        res.add(0.53);
        res.add(0.52);
        List<Integer> res2 = new ArrayList<>();
        res2.add(1);
        res2.add(2);
        res2.add(3);
        res2.add(4);
        filter<Number> numberfilter = number -> number.doubleValue() >  2D;

        List<Double> remove = remove(res, numberfilter);
        List<Integer> remove2 = remove(res2, numberfilter);

    }

    public static <E> List<E> remove (List<E> list,filter<? super E> rule) {
        List<E> toRemove = new ArrayList<>();
        for (E e : list) {
            if(rule.rule(e)){
                toRemove.add(e);
            }
        }
        list.removeAll(toRemove);
        return list;
//    return new ArrayList<>();
    }
}

interface filter<E>{
    boolean rule(E e);
}






