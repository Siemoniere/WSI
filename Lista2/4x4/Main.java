import java.util.*;

public class Main {
    
    public static void main(String[] args) {
        Taquin taquin = new Taquin();
        taquin.makeRandomMoves(20); // Wykonaj do 20 losowych ruchów
        taquin.print();
        if (!taquin.isSolvable()) {
            System.out.println("Układanka nie jest rozwiązywalna.");
            return;
        }
        //kolejka priorytetowa
        PriorityQueue<Taquin> queue = new PriorityQueue<>((a, b) -> 
            Integer.compare(a.getMoves() + a.heuristic(), b.getMoves() + b.heuristic())
        );
        Set<String> visited = new HashSet<>();
        
        // Dodajemy początkowy stan do kolejki
        queue.add(taquin);
        visited.add(Arrays.toString(taquin.getArr()));  // Dodaj początkowy stan do odwiedzonych
        
        while (!queue.isEmpty()) {
            Taquin current = queue.poll();
            
            // Wyświetlamy stan układanki
            current.print();
            
            // Sprawdzamy, czy rozwiązanie zostało znalezione
            if (current.isSolved()) {
                System.out.println("Znaleziono rozwiązanie w " + current.getMoves() + " ruchach.");
                current.print();
                return;
            }
            
            // Dodajemy nowe ruchy do kolejki
            for (Taquin neighbor : current.getNeighbors()) {
                String neighborState = Arrays.toString(neighbor.getArr());
                
                // Jeśli stan nie był jeszcze odwiedzony, dodajemy go do kolejki
                if (!visited.contains(neighborState)) {
                    visited.add(neighborState);
                    queue.add(neighbor);
                }
            }
        }
        
        // Jeśli nie znaleziono rozwiązania
        System.out.println("Nie znaleziono rozwiązania.");
    }
}