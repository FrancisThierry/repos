var express = require('express');
var app = require("https-localhost")()

//setting middleware
app.use(express.static(__dirname + '/ServeFolder')); //Serves resources from public folder


var server = app.listen(5000);