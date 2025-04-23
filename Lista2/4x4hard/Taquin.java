//układanka 4x4

import java.util.ArrayList;

public class Taquin {

    int[] arr = new int[16];
    int moves = 0;
    ArrayList<int[]> history = new ArrayList<>();

    public Taquin() {
        // Tworzymy tablicę liczb od 1 do 15
        int[] temp = new int[16];
        for (int i = 0; i < 15; i++) {
            temp[i] = i + 1;
        }
    
        // Losujemy permutację Fisher-Yates
        java.util.Random random = new java.util.Random();
        for (int i = temp.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int swap = temp[i];
            temp[i] = temp[j];
            temp[j] = swap;
        }
    
        // Przepisujemy wylosowane liczby do tablicy arr
        for (int i = 0; i < 15; i++) {
            arr[i] = temp[i];
        }
    
        // Ostatni element ustawiamy na 0
        arr[15] = 0;
    
        // Dodajemy początkowy stan do historii
        history.add(arr.clone());
    }
    public void print() {
        System.out.println("------------------");
        System.out.println("Liczba ruchów: " + moves);
        System.out.println("Stany w historii: " + history.size());
        System.err.println("------------------");
        for (int i = 0; i < arr.length; i++) {
            System.out.print(arr[i] + " ");
            if ((i + 1) % 4 == 0) {
                System.out.println();
            }
        }
    }
    public int getMoves() {
        return moves;
    }
    public ArrayList<int[]> getHistory() {
        return history;
    }

    public Taquin moveUp() {
        int[] arrCopy = arr.clone();
        int index = 0;
        for (int i = 0; i < arrCopy.length; i++) {
            if (arr[i] == 0) {
                index = i;
            }
        }
        if (index > 3) {
            int temp = arrCopy[index - 4];
            arrCopy[index - 4] = arrCopy[index];
            arrCopy[index] = temp;
            moves++;
            history.add(arrCopy);
        }
        Taquin newTaquin = new Taquin();
        newTaquin.arr = arrCopy;
        newTaquin.moves = moves;
        newTaquin.history = new ArrayList<>(this.history);
        return newTaquin;
    }

    public Taquin moveDown() {
        int[] arrCopy = arr.clone();
        int index = 0;
        for (int i = 0; i < arrCopy.length; i++) {
            if (arr[i] == 0) {
                index = i;
            }
        }
        if (index < 12) {
            int temp = arrCopy[index + 4];
            arrCopy[index + 4] = arrCopy[index];
            arrCopy[index] = temp;
            moves++;
            history.add(arrCopy);
        }
        Taquin newTaquin = new Taquin();
        newTaquin.arr = arrCopy;
        newTaquin.moves = moves;
        newTaquin.history = new ArrayList<>(this.history);
        return newTaquin;        
    }

    public Taquin moveLeft() {
        int[] arrCopy = arr.clone();
        int index = 0;
        for (int i = 0; i < arrCopy.length; i++) {
            if (arr[i] == 0) {
                index = i;
            }
        }
        if (index % 4 != 0) {
            int temp = arrCopy[index - 1];
            arrCopy[index - 1] = arrCopy[index];
            arrCopy[index] = temp;
            moves++;
            history.add(arrCopy);
        }
        Taquin newTaquin = new Taquin();
        newTaquin.arr = arrCopy;
        newTaquin.moves = moves;
        newTaquin.history = new ArrayList<>(this.history);
        return newTaquin;
    }

    public Taquin moveRight() {
        int[] arrCopy = arr.clone();
        int index = 0;
        for (int i = 0; i < arrCopy.length; i++) {
            if (arr[i] == 0) {
                index = i;
            }
        }
        if (index % 4 != 3) {
            int temp = arrCopy[index + 1];
            arrCopy[index + 1] = arrCopy[index];
            arrCopy[index] = temp;
            moves++;
            history.add(arrCopy);
        }
        Taquin newTaquin = new Taquin();
        newTaquin.arr = arrCopy;
        newTaquin.moves = moves;
        newTaquin.history = new ArrayList<>(this.history);
        return newTaquin;
    }

    public boolean isSolved() {
        for (int i = 0; i < arr.length - 1; i++) {
            if (arr[i] != i + 1) {
                return false;
            }
        }
        if (arr[arr.length - 1] != 0) {
            return false;
        }
        return true;
    }

    public int howManyWrong(int[] arr) {
        int count = 0;
        for (int i = 0; i < arr.length - 1; i++) {
            if (arr[i] != i + 1) {
                count++;
            }
        }
        if (arr[arr.length - 1] != 0) {
            count++;
        }
        return count;
    }

    public int manhattanDistance(int[] arr) {
        int distance = 0;
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] != 0) {
                int x = arr[i] % 4;
                int y = arr[i] / 4;
                int targetX = i % 4;
                int targetY = i / 4;
                distance += Math.abs(x - targetX) + Math.abs(y - targetY);
            }
        }
        return distance;
    }
    public int[] getArr() {
        return arr.clone();
    }

    // Generowanie sąsiednich stanów
    public ArrayList<Taquin> getNeighbors() {
        ArrayList<Taquin> neighbors = new ArrayList<>();
        int index = findZeroIndex(); // Indeks pustego pola
        int row = index / 4, col = index % 4;

        // Ruchy: góra, dół, lewo, prawo
        int[][] moves = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        for (int[] move : moves) {
            int newRow = row + move[0];
            int newCol = col + move[1];
            if (newRow >= 0 && newRow < 4 && newCol >= 0 && newCol < 4) {
                int newIndex = newRow * 4 + newCol;
                Taquin neighbor = new Taquin();
                System.arraycopy(arr, 0, neighbor.arr, 0, arr.length);
                // Zamiana pustego pola z sąsiednim kafelkiem
                neighbor.arr[index] = neighbor.arr[newIndex];
                neighbor.arr[newIndex] = 0;
                neighbor.moves = this.moves + 1;
                neighbor.history.add(arr.clone());
                neighbors.add(neighbor);
            }
        }
        return neighbors;
    }
    public boolean isSolvable() {
        int inversions = 0;
        
        // Liczymy liczbę inwersji
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == 0) continue; // Pomijamy puste pole
            for (int j = i + 1; j < arr.length; j++) {
                if (arr[j] != 0 && arr[i] > arr[j]) {
                    inversions++;
                }
            }
        }
    
        // Znajdujemy wiersz pustego pola licząc od dołu
        int emptyRowFromBottom = 4 - (findZeroIndex() / 4);
    
        // Sprawdzamy kryterium rozwiązywalności
        if (inversions % 2 == 0) {
            return emptyRowFromBottom % 2 != 0; // Parzysta liczba inwersji -> nieparzysty wiersz
        } else {
            return emptyRowFromBottom % 2 == 0; // Nieparzysta liczba inwersji -> parzysty wiersz
        }
    }
    
    // Pomocnicza metoda do znalezienia indeksu pustego pola
    private int findZeroIndex() {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == 0) {
                return i;
            }
        }
        return -1; // Nie powinno się zdarzyć
    }
    int heuristic() {
        return manhattanDistance(arr) + howManyWrong(arr);
    }
}