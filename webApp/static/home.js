function getResult() {
    var url = "http://localhost:8000";
    var endpoint = "/result";
    var http = new XMLHttpRequest();
    var fileInput = document.getElementById('imageInput');

    var formData = new FormData();
    formData.append('imageInput', fileInput.files[0]);

    http.open("POST", url + endpoint, true);

    http.onreadystatechange = function () {
        var DONE = 4;
        var OK = 200;

        if (http.readyState === DONE) {
            if (http.status === OK && http.responseText) {
                var replyString = http.responseText;
                var resultObj = JSON.parse(replyString);

                
                document.getElementById("resultLabel").innerHTML = "Predicted Label: " + resultObj.label;
                document.getElementById("resultConf").innerHTML = "Confidence: " + resultObj.conf;
            } else {
                console.error('Error fetching data:', http.status, http.statusText);
            }
        }
    };

    http.send(formData);
}
