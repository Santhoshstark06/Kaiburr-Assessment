
Basic Functionalities :-
Show All servers :-
Displays all the servers if no parameters are passed.When server id is passed as a parameter - return a single server or 404 if there’s no such a server.
   	        var xhttp = new XMLHttpRequest();
   		//formatting data
   		xhttp.open("GET", "http://localhost:8011/restapi/get_servers", true);
   		xhttp.setRequestHeader("Content-type", "application/json"); 
   		xhttp.send();
      
      
      2. #### Add a Server :- The server object is passed as a json-encoded message body. Here’s an example:

   	      	var xhttp = new XMLHttpRequest(); 
   	        var url =   document.url_input.name.value + "-";
   	          url = url + document.url_input.id.value + "-";
   	          url = url + document.url_input.language.value + "-"; 
   	          url = url + document.url_input.framework.value  ;
   	        xhttp.open("PUT", "http://localhost:8011/restapi/get_servers/"+url, true);
   	        xhttp.send();
            
            
            
            
            		{ 
			“name”: ”my centos”,
		 	“id”: “123”,
		  	“language”:”java”,
		   	“framework”:”django” 
		}
    
    
    Delete Server:-
The parameter is a server ID.
   	       var xhttp = new XMLHttpRequest(); 
   	       xhttp.open("DELETE", "http://localhost:8011/restapi/get_servers/"+id, true);
   	       xhttp.send();
           
           Find Servers by Name/ ID :-
The parameter is a string. Checks if a server name contains this string and return one or more servers found. Return 404 if nothing is found.
