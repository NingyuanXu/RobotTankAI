package main;

import java.util.LinkedList;

public class CircularQueue<T> extends LinkedList<T> {
    private int capacity = 10;

    public CircularQueue(int capacity){
        this.capacity = capacity;
    }

    public boolean add(T e) {
        if(size() >= capacity)
            removeFirst();
        return super.add(e);
    }

    public Object [] toArray() {
        return super.toArray();
    }
}