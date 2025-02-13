using System.Security.Claims;
using GameRecommender.Controllers;
using GameRecommender.Data;
using GameRecommender.Models;
using GameRecommender.Services;
using Microsoft.AspNetCore.Mvc;
using Moq;
using NUnit.Framework;
using NUnit.Framework.Legacy;

namespace GameRecommender.Tests
{
    [TestFixture]
    public class UserControllerTests
    {
        private Mock<IDatabaseHandler> _mockDatabaseHandler;
        private UserService _userService;
        private Mock<IConfiguration> _mockConfiguration;
        private UserController _controller;

        [SetUp]
        public void Setup()
        {
            _mockConfiguration = new Mock<IConfiguration>();
            _mockConfiguration.Setup(c => c["Jwt:Secret"]).Returns("TestSecretKey");
            _mockConfiguration.Setup(c => c["Jwt:Issuer"]).Returns("TestIssuer");
            _mockConfiguration.Setup(c => c["Jwt:Audience"]).Returns("TestAudience");

            _mockDatabaseHandler = new Mock<IDatabaseHandler>();
            _userService = new UserService(_mockDatabaseHandler.Object);

            _controller = new UserController(_userService, _mockConfiguration.Object);

            var mockHttpContext = new Mock<HttpContext>();
            var mockClaimsPrincipal = new Mock<ClaimsPrincipal>();
            mockHttpContext.Setup(x => x.User).Returns(mockClaimsPrincipal.Object);
            _controller.ControllerContext = new ControllerContext() { HttpContext = mockHttpContext.Object };
        }

        [Test]
        public async Task Register_ValidUser_ReturnsOkResult()
        {
            // Arrange
            var user = new User(Guid.NewGuid(), "testuser", "test@example.com", "password");
            _mockDatabaseHandler.Setup(service => service.RegisterUser(It.IsAny<User>())).ReturnsAsync(user);

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
            var user = new User(Guid.NewGuid(), "testuser", "test@example.com", "password");
            _mockDatabaseHandler.Setup(service => service.LoginByUsername(userLogin.Username, userLogin.Password)).ReturnsAsync(user);

            // Act
            var result = await _controller.Login(userLogin);

            // Assert
            ClassicAssert.IsInstanceOf<OkObjectResult>(result);
            var okResult = (OkObjectResult)result;
            ClassicAssert.NotNull(okResult.Value);
        }

        [Test]
        public async Task Login_InvalidCredentials_ReturnsUnauthorized()
        {
            // Arrange
            var userLogin = new UserLoginModel { Username = "testuser", Password = "wrongpassword" };
            _mockDatabaseHandler.Setup(service => service.LoginByUsername(userLogin.Username, userLogin.Password)).ReturnsAsync((User)null);

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
            var user = new User(userId, "testuser", "test@example.com", "password");
            _mockDatabaseHandler.Setup(s => s.UpdateUser(It.IsAny<User>())).ReturnsAsync(user);

            _controller.HttpContext.User = new ClaimsPrincipal(new ClaimsIdentity(new[] { new Claim(ClaimTypes.NameIdentifier, userId.ToString()) }));

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
            var user = new User(Guid.NewGuid(), "testuser", "test@example.com", "password");

            _controller.HttpContext.User = new ClaimsPrincipal(new ClaimsIdentity(new[] { new Claim(ClaimTypes.NameIdentifier, userIdFromToken.ToString()) }));

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
            var userToDelete = new User(userId, "testuser", "test@example.com", "password");
            _mockDatabaseHandler.Setup(s => s.DeleteUser(It.IsAny<User>())).ReturnsAsync(true);

            _controller.HttpContext.User = new ClaimsPrincipal(new ClaimsIdentity(new[] { new Claim(ClaimTypes.NameIdentifier, userId.ToString()) }));

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
            _mockDatabaseHandler.Setup(s => s.DeleteUser(It.IsAny<User>())).ReturnsAsync(false);

            _controller.HttpContext.User = new ClaimsPrincipal(new ClaimsIdentity(new[] { new Claim(ClaimTypes.NameIdentifier, userId.ToString()) }));

            // Act
            var result = await _controller.DeleteUser();

            // Assert
            ClassicAssert.IsInstanceOf<NotFoundObjectResult>(result);
        }

        [Test]
        public async Task DeleteUser_InvalidUserIdInToken_ReturnsBadRequest()
        {
            // Arrange
            _controller.HttpContext.User = new ClaimsPrincipal(new ClaimsIdentity(new[] { new Claim(ClaimTypes.NameIdentifier, "invalid-guid") }));

            // Act
            var result = await _controller.DeleteUser();

            // Assert
            ClassicAssert.IsInstanceOf<BadRequestObjectResult>(result);
        }
    }
}