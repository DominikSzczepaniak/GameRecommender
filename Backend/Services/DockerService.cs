using System.Runtime.InteropServices;
using GameRecommender.Interfaces;

namespace GameRecommender.Services;
using Docker.DotNet;
using Docker.DotNet.Models;
using System;
using System.Threading.Tasks;

public class DockerService : IDockerRunner
{
    private readonly DockerClient _dockerClient;

    public DockerService()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            _dockerClient = new DockerClientConfiguration(new Uri("npipe://./pipe/docker_engine"))
                .CreateClient();
            return;
        }
        _dockerClient = new DockerClientConfiguration(new Uri("unix:///var/run/docker.sock"))
            .CreateClient();
    }

    public async Task<(string host, string containerId)> StartContainerWithFreePort(string imageName)
    {
        int freePort = PortFinder.GetAvailablePort();

        var response = await _dockerClient.Containers.CreateContainerAsync(new CreateContainerParameters
        {
            Image = imageName,
            HostConfig = new HostConfig
            {
                PortBindings = new Dictionary<string, IList<PortBinding>>
                {
                    { "5000/tcp", new List<PortBinding> { new PortBinding { HostPort = freePort.ToString() } } }
                }
            }
        });

        await _dockerClient.Containers.StartContainerAsync(response.ID, new ContainerStartParameters());
        return ($"http://localhost:{freePort}", response.ID);
    }

    public async Task StopContainer(string containerId)
    {
        await _dockerClient.Containers.StopContainerAsync(containerId, new ContainerStopParameters());
    }

    public async Task<List<string>> GetRecommendations(Guid userId)
    {
        return new List<string>();
    }
}