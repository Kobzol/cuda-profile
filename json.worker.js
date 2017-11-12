self.onmessage = function(message)
{
    self.postMessage(JSON.parse(message.data));
};
