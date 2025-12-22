<template>
  <div>
    <header>
      <h1>Нобелевские премии</h1>
    </header>

    <nav>
      <section>
        <router-link to="/laureats" active-class="active">Лауреаты</router-link>
        <router-link to="/prizes" active-class="active">Премии</router-link>
      </section>
    </nav>

    <main>
      <router-view></router-view>
    </main>

    <footer>
      <p>&copy; 2025 Информация о Нобелевских премиях</p>
    </footer>
  </div>
</template>

<script>
export default {
  name: 'App'
}
</script>




