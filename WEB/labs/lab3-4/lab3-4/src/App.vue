<template>
  <header>
    <h1>Нобелевские премии</h1>
    <nav>
      <RouterLink to="/laureates">Лауреаты</RouterLink> |
      <RouterLink to="/prizes">Премии</RouterLink>
    </nav>
  </header>

  <main>
    <RouterView />
  </main>

  <footer>
    <p>ЛР-3. 2025</p>
  </footer>
</template>

<style>
header {
  background: #d9d9d9;
  padding: 8px;
  text-align: center;
}

nav {
  background: #161616;
  padding: 8px;
}

nav a {
  color: #f2f2f2;
  text-decoration: none;
  margin: 0 5px;
}

nav a:hover {
  color: #ff6600;
}

footer {
  background: #000;
  color: white;
  text-align: center;
  padding: 8px;
}
</style>