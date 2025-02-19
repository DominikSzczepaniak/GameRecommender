using System.Runtime.InteropServices;
using GameRecommender.Interfaces;
using GameRecommender.Models;

namespace GameRecommender.Services;
using Docker.DotNet;
using Docker.DotNet.Models;
using System;
using System.Threading.Tasks;

public class DockerService : IDockerRunner
{
    private readonly DockerClient _dockerClient;
    private readonly IRecommenderApiService _recommenderApiService;
    private readonly Dictionary<int, string> _engineNumberToImageName = new Dictionary<int, string>
    {
        { 1, "lightFM" }
    };

    public DockerService(IRecommenderApiService recommenderApiService)
    {
        _recommenderApiService = recommenderApiService;
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

    public Task StopContainer(string containerId)
    {
        return _dockerClient.Containers.StopContainerAsync(containerId, new ContainerStopParameters());
    }

    public async Task<List<string>> GetRecommendations(Guid userId, int numberOfRecommendations, int engineNumber)
    {
        if (!_engineNumberToImageName.ContainsKey(engineNumber))
        {
            throw new ArgumentException("No such engine");
        }
        var (host, containerId) = await StartContainerWithFreePort(_engineNumberToImageName[engineNumber]);
        var result = await _recommenderApiService.GetGameList(userId, numberOfRecommendations, host);
        StopContainer(containerId);
        return result;
    }

    public async Task<bool> LearnUser(List<UserGameDao> games, int engineNumber)
    {
        if (!_engineNumberToImageName.ContainsKey(engineNumber))
        {
            throw new ArgumentException("No such engine");
        }
        var (host, containerId) = await StartContainerWithFreePort(_engineNumberToImageName[engineNumber]);
        var result = await _recommenderApiService.LearnUser(games, host);
        StopContainer(containerId);
        return result;
    }
}