using System;
using System.Collections.Generic;
using System.Security.Claims;
using System.Threading.Tasks;
using GameRecommender.Controllers;
using GameRecommender.Interfaces;
using GameRecommender.Models;
using GameRecommender.Services;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Configuration;
using Moq;
using NUnit.Framework;
using NUnit.Framework.Legacy;

namespace GameRecommender.Tests
{
    [TestFixture]
    public class UserControllerTests
    {
        private Mock<UserService> _mockUserService;
        private Mock<IConfiguration> _mockConfiguration;
        private UserController _controller;

        [SetUp]
        public void Setup()
        {
            _mockUserService = new Mock<UserService>(MockBehavior.Strict);
            _mockConfiguration = new Mock<IConfiguration>();
            _mockConfiguration.Setup(c => c["Jwt:Secret"]).Returns("TestSecretKey");
            _mockConfiguration.Setup(c => c["Jwt:Issuer"]).Returns("TestIssuer");
            _mockConfiguration.Setup(c => c["Jwt:Audience"]).Returns("TestAudience");

            _controller = new UserController(_mockUserService.Object, _mockConfiguration.Object);

            // Mock HttpContext for Authorization
            var mockHttpContext = new Mock<HttpContext>();
            var mockClaimsPrincipal = new Mock<ClaimsPrincipal>();
            mockHttpContext.Setup(x => x.User).Returns(mockClaimsPrincipal.Object);
            _controller.ControllerContext = new ControllerContext() { HttpContext = mockHttpContext.Object };
        }

        [Test]
        public async Task Register_ValidUser_ReturnsOkResult()
        {
            // Arrange
            var user = new User(Guid.NewGuid(), "testuser", "test@example.com", "password", new List<int>(), new Dictionary<int, bool>());
            _mockUserService.Setup(service => service.RegisterUser(user)).ReturnsAsync(user);

            // Act
            var result = await _controller.Register(user);

            // Assert
            ClassicAssert.IsInstanceOf<OkObjectResult>(result);
            var okResult = (OkObjectResult)result;
            ClassicAssert.AreEqual(user, okResult.Value);
        }

        [Test]
        public async Task Login_ValidCredentials_ReturnsOkResultWithToken()
        {
            // Arrange
            var userLogin = new UserLoginModel { Username = "testuser", Password = "password" };
            var user = new User(Guid.NewGuid(), "testuser", "test@example.com", "password", new List<int>(), new Dictionary<int, bool>());
            _mockUserService.Setup(service => service.LoginByUsername(userLogin.Username, userLogin.Password)).ReturnsAsync(user);

            // Act
            var result = await _controller.Login(userLogin);

            // Assert
            ClassicAssert.IsInstanceOf<OkObjectResult>(result);
            var okResult = (OkObjectResult)result;
            ClassicAssert.NotNull(okResult.Value);
            ClassicAssert.IsInstanceOf<object>(okResult.Value); // Check if it's an anonymous object with a "token" property
        }

        [Test]
        public async Task Login_InvalidCredentials_ReturnsUnauthorized()
        {
            // Arrange
            var userLogin = new UserLoginModel { Username = "testuser", Password = "wrongpassword" };
            _mockUserService.Setup(service => service.LoginByUsername(userLogin.Username, userLogin.Password)).ReturnsAsync((User)null);

            // Act
            var result = await _controller.Login(userLogin);

            // Assert
            ClassicAssert.IsInstanceOf<UnauthorizedResult>(result);
        }

        [Test]
        public async Task UpdateUser_ValidInput_ReturnsOk()
        {
            // Arrange
            var userId = Guid.NewGuid();
            var user = new User(userId, "testuser", "test@example.com", "password", new List<int>(), new Dictionary<int, bool>());
            _mockUserService.Setup(s => s.UpdateUser(user)).ReturnsAsync(user);
            _controller.HttpContext.User.AddIdentity(new ClaimsIdentity(new[] { new Claim(ClaimTypes.NameIdentifier, userId.ToString()) }));

            // Act
            var result = await _controller.UpdateUser(user);

            // Assert
            ClassicAssert.IsInstanceOf<OkObjectResult>(result);
        }

        [Test]
        public async Task UpdateUser_MismatchUserId_ReturnsForbid()
        {
            // Arrange
            var userIdFromToken = Guid.NewGuid();
            var user = new User(Guid.NewGuid(), "testuser", "test@example.com", "password", new List<int>(), new Dictionary<int, bool>());
            _controller.HttpContext.User.AddIdentity(new ClaimsIdentity(new[] { new Claim(ClaimTypes.NameIdentifier, userIdFromToken.ToString()) }));

            // Act
            var result = await _controller.UpdateUser(user);

            // Assert
            ClassicAssert.IsInstanceOf<ForbidResult>(result);
        }

        [Test]
        public async Task DeleteUser_ValidInput_ReturnsOk()
        {
            // Arrange
            var userId = Guid.NewGuid();
            _mockUserService.Setup(s => s.DeleteUser(It.IsAny<User>())).ReturnsAsync(true); // Any User object as it's only using the Id
            _controller.HttpContext.User.AddIdentity(new ClaimsIdentity(new[] { new Claim(ClaimTypes.NameIdentifier, userId.ToString()) }));

            // Act
            var result = await _controller.DeleteUser();

            // Assert
            ClassicAssert.IsInstanceOf<OkObjectResult>(result);
        }

        [Test]
        public async Task DeleteUser_UserNotFound_ReturnsNotFound()
        {
            // Arrange
            var userId = Guid.NewGuid();
            _mockUserService.Setup(s => s.DeleteUser(It.IsAny<User>())).ReturnsAsync(false);
            _controller.HttpContext.User.AddIdentity(new ClaimsIdentity(new[] { new Claim(ClaimTypes.NameIdentifier, userId.ToString()) }));

            // Act
            var result = await _controller.DeleteUser();

            // Assert
            ClassicAssert.IsInstanceOf<NotFoundObjectResult>(result);
        }

        [Test]
        public async Task DeleteUser_InvalidUserIdInToken_ReturnsBadRequest()
        {
            // Arrange
            _controller.HttpContext.User.AddIdentity(new ClaimsIdentity(new[] { new Claim(ClaimTypes.NameIdentifier, "invalid-guid") }));

            // Act
            var result = await _controller.DeleteUser();

            // Assert
            ClassicAssert.IsInstanceOf<BadRequestObjectResult>(result);
        }
    }
}