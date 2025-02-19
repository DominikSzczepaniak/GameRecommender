using System;
using System.Net;
using System.Net.Sockets;
namespace GameRecommender.Services;
public static class PortFinder
{
    public static int GetAvailablePort()
    {
        using (var socket = new TcpListener(IPAddress.Loopback, 0))
        {
            socket.Start();
            return ((IPEndPoint)socket.LocalEndpoint).Port;
        }
    }
}