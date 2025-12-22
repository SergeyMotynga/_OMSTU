<template>
  <TableComponent
    title="Список лауреатов"
    :headers="headers"
    :data="laureatsData"
  />
</template>

<script>
import TableComponent from '../components/TableComponent.vue'

export default {
  name: 'LaureatsView',
  components: {
    TableComponent
  },
  data() {
    return {
      headers: ['Имя / Название', 'Дата рождения / Основания', 'Число премий'],
      laureatsData: [
        { name: 'Альберт Эйнштейн', date: '14.03.1879', prizes: '1' },
        { name: 'Мария Кюри', date: '07.11.1867', prizes: '2' },
        { name: 'Международный Красный Крест', date: '1863', prizes: '3' },
        { name: 'Международный Красный Крест', date: '1863', prizes: '3' },
        { name: 'Международный Красный Крест', date: '1863', prizes: '3' },
        { name: 'Международный Красный Крест', date: '1863', prizes: '3' }
      ]
    }
  }
}
</script>




