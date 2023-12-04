function getResult() {
    var url = "http://localhost:8000";   // The URL and the port number must match server-side
    var endpoint = "/result";            // Endpoint must match server endpoint
    var http = new XMLHttpRequest();
    var fileInput = document.getElementById('imageInput')

    http.open("GET", url+endpoint, true);
    http.onreadystatechange = function() {
        var DONE = 4;
        var OK = 200;
        if (http.readyState == DONE && http.status == OK && http.responseText) {
            var replyString = http.responseText;

            document.getElementById("result").innerHTML = "JSON received: " + replyString;
            document.getElementById("result").innerHTML += "<br>";
        }
    };

    http.send();
}