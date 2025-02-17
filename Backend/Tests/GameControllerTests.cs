using System;
using System.Security.Claims;
using System.Threading.Tasks;
using GameRecommender.Controllers;
using GameRecommender.Interfaces;
using GameRecommender.Models;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Moq;
using NUnit.Framework;
using NUnit.Framework.Legacy;

namespace GameRecommender.Tests;

[TestFixture]
public class GameControllerTests
{
    private Mock<IGameService> _mockGameService;
    private GameController _controller;

    [SetUp]
    public void SetUp()
    {
        _mockGameService = new Mock<IGameService>();
        _controller = new GameController(_mockGameService.Object);
    }

    private void SetUserInContext(string userId)
    {
        var userClaims = new ClaimsPrincipal(new ClaimsIdentity(new Claim[]
        {
            new Claim(ClaimTypes.NameIdentifier, userId)
        }, "mock"));

        _controller.ControllerContext = new ControllerContext
        {
            HttpContext = new DefaultHttpContext { User = userClaims }
        };
    }

    [Test]
    public async Task AddOpinionForUserAndGame_ValidRequest_ReturnsOk()
    {
        // Arrange
        var userId = Guid.NewGuid();
        SetUserInContext(userId.ToString());

        var request = new AddOpinionRequest
        {
            user = new User(userId, "test", "test", "test"),
            gameDto = new UserGameDto("1", true)
        };

        _mockGameService.Setup(service => service.AddOpinionForUserAndGame(userId, It.IsAny<UserGameLogic>())).Returns(Task.CompletedTask);

        // Act
        var result = await _controller.AddOpinionForUserAndGame(request);

        // Assert
        ClassicAssert.IsInstanceOf<OkResult>(result);
    }

    [Test]
    public async Task AddOpinionForUserAndGame_InvalidUserIdInToken_ReturnsBadRequest()
    {
        // Arrange
        SetUserInContext("invalid-guid");
        var request = new AddOpinionRequest
        {
            user = new User(new Guid(), "test", "test", "test"),
            gameDto = new UserGameDto("1", true)
        };

        // Act
        var result = await _controller.AddOpinionForUserAndGame(request);

        // Assert
        ClassicAssert.IsInstanceOf<BadRequestObjectResult>(result);
        var badRequestResult = (BadRequestObjectResult)result;
        ClassicAssert.AreEqual("Invalid user ID in token.", badRequestResult.Value);
    }

    [Test]
    public async Task AddOpinionForUserAndGame_MismatchedUserId_ReturnsForbid()
    {
        // Arrange
        var userIdFromToken = Guid.NewGuid();
        var requestUserId = Guid.NewGuid();
        SetUserInContext(userIdFromToken.ToString());

        var request = new AddOpinionRequest
        {
            user = new User(requestUserId, "test", "test", "test"),
            gameDto = new UserGameDto("1", true)
        };

        // Act
        var result = await _controller.AddOpinionForUserAndGame(request);

        // Assert
        ClassicAssert.IsInstanceOf<ForbidResult>(result);
    }
}