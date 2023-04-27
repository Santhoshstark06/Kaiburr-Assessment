public class Server {
    private int id;
    private String name;
    private String ipAddress;
    
    // constructor, getters and setters
}
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ServerService {
    private Map<Integer, Server> servers = new HashMap<>();
    private int nextId = 1;
    
    public List<Server> getAllServers() {
        return new ArrayList<>(servers.values());
    }
    
    public Server getServer(int id) {
        return servers.get(id);
    }
    
    public Server createServer(Server server) {
        server.setId(nextId);
        servers.put(nextId, server);
        nextId++;
        return server;
    }
    
    public void deleteServer(int id) {
        servers.remove(id);
    }
}

//creating a ServerController class to handle our REST API endpoints
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
public class ServerController {
    private ServerService serverService = new ServerService();
    
    @GetMapping("/servers")
    public ResponseEntity<Object> getAllServers() {
        return new ResponseEntity<>(serverService.getAllServers(), HttpStatus.OK);
    }
    
    @GetMapping("/servers/{id}")
    public ResponseEntity<Object> getServer(@PathVariable int id) {
        Server server = serverService.getServer(id);
        if (server == null) {
            return new ResponseEntity<>("Server not found", HttpStatus.NOT_FOUND);
        }
        return new ResponseEntity<>(server, HttpStatus.OK);
    }
    
    @PutMapping("/servers")
    public ResponseEntity<Object> createServer(@RequestBody Server server) {
        Server createdServer = serverService.createServer(server);
        return new ResponseEntity<>(createdServer, HttpStatus.CREATED);
    }
    
    @DeleteMapping("/servers/{id}")
    public ResponseEntity<Object> deleteServer(@PathVariable int id) {
        serverService.deleteServer(id);
        return new ResponseEntity<>(HttpStatus.NO_CONTENT);
    }
}
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class Application {
    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }
}
//With these classes, you can run a Java application that provides a REST API with endpoints for searching, creating and deleting server objects
